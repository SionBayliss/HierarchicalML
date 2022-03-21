#!/usr/bin/env perl

use strict;
use warnings;
use Getopt::Long qw(GetOptions);
use Pod::Usage;
use Cwd 'abs_path';
use File::Basename;

# accept assemblies or read files and filter/correct reads, create unitigs, map, assemble, annotate and qc sample.

=head1  SYNOPSIS

 wgs-pipe.pl [--input /path/to/input_list][--runid int] [--ref /path/to/reference_genome] [--output /path/to/output_directory] [opt_args]
 
 Input options:
 --output		path to output directory [required]
 --sample-id		output prefix [required]
 --R1 			input pe fastq 1 [optional]
 --R2 			input pe fastq 2 [optional]
 --contigs		input contig file [optional]
 --remote		remote ENA/SRA accession [optional]		
 [required: either (--R1 and --R2), --contigs or --remote]

 Options:
 --ref			reference genome for comparison 
 			in fasta/gbk/gff format [optional]
 --readqc-options	options passed to read qc [optional]
 --unitig-options	options passed to unitig creation [optional]
 
 Module options:
 --no-unitigs		switch off unitig module
 --no-read-qc		switch off read qc module
 
 Test options:
 --test-reads		run pipeline on test reads
 --test-ctgs		run pipeline on test contigs

 General options:
 --cpus			cpu threads [default: 1]
 --force		force assembly/mapping over previous files 
 			[default: off]
 --keep			keep intermediate files [0 - none (default),
 				1 - qc-reads only, 2 - all]
 --help			usage information
 
=cut

# path to executing script
my $script_path = abs_path(dirname($0));

# check dependencies and switch off sections of the pipeline accordingly.

# command line options
my $R1 = "";
my $R2 = "";
my $contigs = "";
my $remote = "";

my $output = '';

my $sample_id = "";
my $ref = "";

my $read_qc_off = 0;
my $unitigs_off = 0;

my $read_options = "";
my $unitig_options = "";

my $test_reads = 0;
my $test_ctgs = 0;

my $cpus = 1;
my $keep = 0;
my $force = 0;
my $help = 0;

GetOptions(

	'R1=s' => \$R1,
	'R2=s' => \$R2,
	'contigs=s' => \$contigs,
	'remote=s' => \$remote,
	
	'sample-id=s' => \$sample_id,
	'output=s' 	=> \$output,
	
	'ref=s' => \$ref,
	
	'readqc-options=s' => \$read_options,
	'unitig-options=s' => \$unitig_options,
	
	'no-readqc' => \$read_qc_off,
	'no-unitigs' => \$unitigs_off,
	
	'test-reads' => \$test_reads,
	'test-ctgs' => \$test_ctgs,
	
	'cpus=i' => \$cpus,
	'keep=i' => \$keep,
	'force' => \$force,
	'help|?' => \$help,
				
) or pod2usage(1);
pod2usage(1) if $help;

# feedback
print "\n\n\----------------------------------------------------------\n\n";

# [optional] set test paths
$test_ctgs = 0 if (($test_reads == 1) && ($test_ctgs == 1)); # just test reads if both selected.
if ( ($test_reads == 1) || ($test_ctgs == 1) ){
	
	# feedback 
	print "Setting test options:\n";
	
	# specific test options
	if ( $test_reads == 1 ){
		$R1 = "$script_path/../test/reads_R1.fastq.gz"; 
		$R2 = "$script_path/../test/reads_R2.fastq.gz";
		$ref = "$script_path/../test/example.gbk";
	}else{
		$contigs = "$script_path/../test/example.fna";
		$read_qc_off = 1;
		$ref = "$script_path/../test/example.gbk";
	}
	
	# general options
	$sample_id = "test";
	$force = 1;
	$output = "$script_path/../test/test";
	
	# feedback
	print " - complete\n\n\----------------------------------------------------------\n\n";

}

# check inputs
pod2usage( {-message => q{output path is a required arguement}, -exitval => 1, -verbose => 1 } ) if $output eq ''; 
pod2usage( {-message => q{sample id a required arguement}, -exitval => 1, -verbose => 1 } ) if $sample_id eq ''; 

# check correct combination of inputs have been provided
die " - ERROR: provide inputs with --R1/--R2, --contigs or --remote" unless ( ($contigs ne "") || (($R1 ne "") && ($R2 ne "")) || ($remote ne "") );

# check for multiple inputs
my $in_count = 0;
++$in_count if ($contigs ne "");
++$in_count if (($R1 ne "") && ($R1 ne ""));
++$in_count if ($remote ne "");
die " - ERROR: multiple inputs provided, pick one\n" if $in_count>1;

# check output directory exists
unless ( -e $output ){
	die " - ERROR: could not make output directory ($output)\n" unless mkdir($output); 
} 

# check files exist
die " - ERROR: could not find $contigs\n" if ( ($contigs ne "") && !( -f $contigs ) );
die " - ERROR: could not find $R1\n" if ( ($R1 ne "") && !( -f $R1 ) );
die " - ERROR: could not find $R2\n" if ( ($R2 ne "") && !( -f $R2 ) );
die " - ERROR: could not find $ref\n" if ( ($ref ne "") && !( -f $ref ) );

# set abs paths
$output = abs_path($output);
$ref = abs_path($ref) if $ref ne ""; 
$R1 = abs_path($R1) if $R1 ne "";
$R2 = abs_path($R2) if $R2 ne "";
$contigs = abs_path($contigs) if $contigs ne "";

# check which pipelines to run. 
my @pipelines = ("read-qc","unitigs","mv-contigs");
my %process = ();
for (@pipelines) { $process{$_} = 1 };

# [optional] switch off sections of the pipeline. 

# contigs 
$process{"mv-contigs"} = 0 if $contigs eq "";

# reads
$process{"read-qc"} = 0 if $contigs ne "";   

# reference 
$process{"mapping"} = 0 if $ref eq ""; 

# toggled off
$process{"read-qc"} = 0 if $read_qc_off == 1;  
$process{"unitigs"} = 0 if $unitigs_off == 1;  

# download remote reads
my $read_dir = "$output/remote-reads"; 

# check if files files exist and are non-empty
my $download_reads_check = 1;
if ( $remote ne "" ){

	if ( (-f "$read_dir/$sample_id\_1.fastq.gz") && (-f "$read_dir/$sample_id\_2.fastq.gz")  ){
		print " WARNING: read files exist for $sample_id ";
		if ( !(-z "$read_dir/$sample_id\_1.fastq.gz") && !(-z "$read_dir/$sample_id\_2.fastq.gz")  ){
			print "and are not empty - only downloading if --force\n\n";
			$download_reads_check = 0;
		}else{
			print "and are empty - redownloading\n\n";
		}
	}
}

# download if appropriate
if ( (($remote ne "") && ( $download_reads_check == 1)) || ( ($remote ne "") && ($force == 1) ) ){
	
	# feedback
	print "Downloading Reads:\n";
	
	# check output directory exists
	unless ( -e $read_dir ){
		die " - ERROR: could not make output directory ($read_dir)\n" unless mkdir($read_dir); 
	} 
	$read_dir = abs_path($read_dir);
	
	# make remote command 
	my $remote_command = "perl $script_path/remote_download.pl --name $sample_id --accession $remote --output $read_dir";
			
	# download if appropriate
      if ( (-e $read_dir) || (mkdir($read_dir)) ){
    
    		# store command 
		print " - command = ($remote_command)\n";
		
		# run command
		system("$remote_command");
		die " - ERROR: failed to download reads for $sample_id.\n" if $?;
		
	}else{
	
		die " - WARNING: could not make $read_dir - download not performed\n";
	
	}
	
	# set as working files
	$R1 = abs_path("$read_dir/$sample_id\_1.fastq.gz");
	$R2 = abs_path("$read_dir/$sample_id\_2.fastq.gz");
		
	# feedback 
	print " - complete\n\n\----------------------------------------------------------\n\n";

}elsif($remote ne ""){
	
	# set as working files
	$R1 = abs_path("$read_dir/$sample_id\_1.fastq.gz");
	$R2 = abs_path("$read_dir/$sample_id\_2.fastq.gz");

}

# check working files exist
unless ( (( -f $R1 ) && ( -f $R2 )) || ($contigs ne "") ){
	die " - ERROR: could not find both sets of reads files to process $sample_id\n"; 
}

# perform read qc
if ($process{"read-qc"} == 1){
	
	# feedback
	print "Read QC:\n";
	
	my $read_qc_dir = "$output/read-qc"; 
	
	# check output directory exists
	unless ( -e $read_qc_dir ){
		die " - ERROR: could not make output directory ($read_qc_dir)\n" unless mkdir($read_qc_dir); 
	} 
	$read_qc_dir = abs_path($read_qc_dir);
	
	# set read-qc options
	my @read_opts = split(/\s+/, $read_options);
	push(@read_opts, "--ref $ref") if $ref ne ""; 
	push(@read_opts, "--keep") if $keep == 2; 
	my $read_in = join( " ", @read_opts);
	
	# make read-qc command 
	my $read_command = "perl $script_path/qc-pipe.pl --name $sample_id --R1 $R1 --R2 $R2 --threads $cpus --output $read_qc_dir $read_in";
			
	# run read-qc if appropriate
	if ( (-f "$read_qc_dir/$sample_id\_1.fastq.gz") && ($force == 0) ){
	
		print " - read files already exist (overwrite with -f)\n";
		
     }elsif ( (-e $read_qc_dir) || (mkdir($read_qc_dir)) ){
    
    		# store command 
		print " - command = ($read_command)\n";
		
		# run command
		system("$read_command");
		die " - ERROR: read QC failed for $sample_id.\n" if $?;
		
	}else{
	
		print " - WARNING: could not make $read_qc_dir - read qc not performed\n";
	
	}
		
	# feedback 
	print " - complete\n\n\----------------------------------------------------------\n\n";
}

# check for filtered files - make working if they exist
if ( (-f "$output/read-qc/$sample_id\_1.fastq.gz") && ( -f "$output/read-qc/$sample_id\_2.fastq.gz" ) ){
	$R1 = "$output/read-qc/$sample_id\_1.fastq.gz";
	$R2 = "$output/read-qc/$sample_id\_2.fastq.gz";
}

# create unitigs and count kmers
if ($process{"unitigs"} == 1){

	# feedback
	print "Kmer Counting/Unitigs:\n";
	
	my $kmer_dir = "$output/unitigs"; 
	
	# check output directory exists
	unless ( -e $kmer_dir ){
		die " - ERROR: could not make output directory ($kmer_dir)\n" unless mkdir($kmer_dir); 
	} 
	$kmer_dir = abs_path($kmer_dir);
	
	# set unitig options
	my @uni_opts = ();
	push(@uni_opts, "-z") if $keep == 2; 
	push(@uni_opts, "--R1 $R1 --R2 $R2") if ($R1 ne "");
	push(@uni_opts, "--input $contigs") if ($contigs ne "");
	push(@uni_opts, "--bcalm-opts \"$unitig_options\"") if ($unitig_options ne "");
	my $uni_in = join( " ", @uni_opts);
	
	# make unitig command 
	my $uni_command = "perl $script_path/create_unitigs.pl --name $sample_id --threads $cpus --output $kmer_dir -f $uni_in";
	
	# run kmer counting/unitigs if appropriate
	if ( (-f "$kmer_dir/$sample_id.unitigs.fa.gz") && ($force == 0) ){
	
		print " - unitigs already exist (overwrite with -f)\n";
		
    }elsif ( (-e $kmer_dir) || (mkdir($kmer_dir)) ){
    
    		# store command 
		print " - command = ($uni_command)\n";
		
		# run command
		system("$uni_command");
		
	}else{
	
		print " - WARNING: could not make $kmer_dir - read qc not performed\n";
	
	}
		
	# feedback 
	print " - complete\n\n\----------------------------------------------------------\n\n";
}

# remove filtered/trimmed read files
if ( $keep == 0 ){
	unlink "$output/read-qc/$sample_id\_1.fastq.gz" if ( -f "$output/read-qc/$sample_id\_1.fastq.gz" );
	unlink "$output/read-qc/$sample_id\_2.fastq.gz" if ( -f "$output/read-qc/$sample_id\_2.fastq.gz" );
}

exit
