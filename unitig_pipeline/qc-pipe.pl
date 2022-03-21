#!/usr/bin/env perl

use strict;
use warnings;
use Getopt::Long qw(GetOptions);
use Pod::Usage;
use Cwd 'abs_path';
use File::Basename;

# run quality control and down-sample fastq files

=head1  SYNOPSIS

 qc-pipe -i /path/to/input_fastq -o /path/to/output_directory/
 
 Input/Output:
 --R1		path to read pair 1 [required with --R2]
 --R2		path to read pair 2 [required with --R1]
 --input	path to input file [required unless --R1 and --R2]
 --output	output directory [required]
 --name		sample name before sequencing prefix [required]
 --regex	regular expression for file extention 
 		[default:"_[12].fastq"]
 
 Coverage options:
 --gsize	genome size estimate, required for downsampling [optional]
 --cov		coverage to downsample to after trimming
 		[requires -g, default:100]
 --upper	coverage reduction before trimming 
 		[requires -g, default: 150]
 		
 Trimming/Filtering options:
 --trim-opts	trimmomatic parameters [default: ILLUMINACLIP:*:2:30:10 
 		MINLEN:36 SLIDINGWINDOW:4:20 TOPHRED64]
 --no-trim 	do not trim using trimmomatic
 --musket-ks	K-mers for musket, multiple comma separated [default: 31]
 --musket-opts	Musket command line options
 --no-musket 	do not correct using musket
 
 General options:
 --keep		keep intermediate files [default: off]
 --threads		number of threads [default: 1]
 --test		run qc-pipe test
 --help		usage information
 
=cut

# check for dependencies in path
my $java = 0 ;
$java = 1 if `command -v java;`;
my $fastqc = 0;
$fastqc = 1 if `command -v fastqc;`;
if ( ($java == 0) || ($fastqc==0) ){
	print " - ERROR: java not in path.\n" if $java == 0;
	print " - ERROR: fastqc not in path.\n" if $fastqc == 0;
	die " - ERROR: qc-pipe dependencies missing\n";
}

# script path
my $script_path = abs_path(dirname($0));

# command line options
my $R1 = '';
my $R2 = '';
my $input = '';
my $sample = '';
my $output_dir = '';

my $ref = '';
my $genome_size = 0;
my $coverage = 100;
my $upper = 150;
my $trim_opts = "ILLUMINACLIP:$script_path/PE_All.fasta:2:30:10:2:keepBothReads LEADING:3 TRAILING:3 SLIDINGWINDOW:4:20 MINLEN:36";
my $regex = "_[12].fastq.gz";

my $musket_ks = 31;
my $musket_opts = "";

my $keep_intermediate = 0;
my $test = 0;
my $cores = 1;
my $no_trim = 0;
my $no_musket = 0;
my $help = 0;

GetOptions(

	'help|?' 	=> \$help,
	
	'R1=s' => \$R1,
	'R2=s' => \$R2,
	'input=s' 	=> \$input,
	'name=s' 	=> \$sample,
	'output=s'	=> \$output_dir,
	
	'ref=s' => \$ref, 
	'genome_size=i' => \$genome_size,
	'coverage=i' => \$coverage,
	'upper=i' => \$upper,
	'regex=s' => \$regex,
	
	'trim-opts=s' => \$trim_opts,
	
	'musket-ks=s' => \$musket_ks,
	'musket-opts=s' => \$musket_opts,
	
	'threads=i' => \$cores,
	'test' => \$test,
	'keep' => \$keep_intermediate,
	
	'no-trim' => \$no_trim,
	'no-musket' => \$no_musket,
			
) or pod2usage(1);
pod2usage(1) if $help;

# run test - parameters 
if ( $test == 1 ){
	$input = "$script_path/../test/";
	$output_dir = "$script_path/../test/test/";
	$sample = "reads";
	$regex = "_R[12].fastq.gz";

	$genome_size = 300000;
	$coverage = 50;
	$upper = 80;
}

# check mandatory inputs
pod2usage( {-message => q{sample name is a required argument}, -exitval => 1, -verbose => 1 } ) if $sample eq ''; 
pod2usage( {-message => q{output directory is a required argument}, -exitval => 1, -verbose => 1 } ) if $output_dir eq ''; 

# check inputs
unless ( ($input ne '') || ( ($R1 ne "") && ($R2 ne "") ) ){
	pod2usage( {-message => q{specify --input or --R1 plus --R2}, -exitval => 1, -verbose => 1 } );
}
if ( ($input ne '') && ( ($R1 ne "") && ($R2 ne "") ) ){
	pod2usage( {-message => q{specify either --input or --R1 plus --R2}, -exitval => 1, -verbose => 1 } );
}

# check output directory exists
unless( -e $output_dir ){ die " - ERROR: output directory does not exist and cannot be created\n" unless mkdir($output_dir)};

# [optional] get genome size from reference file
if ( $ref ne "" ){
	
	$ref = abs_path($ref);
	
	if ( -f $ref ){
	
		# genbank file - parse on locus info
		if ( ($ref =~ /.gbk$/) || ($ref =~ /.gb$/) ){
				
			open R, $ref or die " - ERROR: could not open reference file\n";
			while (<R>){
		
				if ( (/^LOCUS/) && (/(\d+)\s+bp/) ){
					$genome_size += $1;
				}		
			}			
		}
		
		# gff file - parse on bp after ##FASTA header
		elsif ( $ref =~ /.gff$/ ){
		
			my $start = 0;
		
			open R, $ref or die " - ERROR: could not open reference file\n";
			while (<R>){
		
				my $line = $_;
				chomp $line;
				
				if (/^##FASTA/){
					$start = 1;
				}
				
				unless ( (/^>/) && ($start == 1) ){
					$genome_size = $genome_size + length($line);
				}		
			}			
		}
		
		# fasta file - parse on non-header lines
		elsif ( ($ref =~ /.fasta$/) || ($ref =~ /.fa$/) || ($ref =~ /.fna$/) || ($ref =~ /.fas1$/) ){
				
			open R, $ref or die " - ERROR: could not open reference file\n";
			while (<R>){
		
				my $line = $_;
				chomp $line;
				
				unless ( $line =~ /^>/ ){
					$genome_size = $genome_size + length($line);
				}		
			}			
		}
		
		# reference format not recognised
		else{
			print " - WARNING: reference format not recognised\n";
		}
		
	}
} 
# make log file
my $log_file = "$output_dir/log.file";
open LOG, ">$log_file" or die " - ERROR: could not make log file\n";

# find pe-1 and pe-2
die " - ERROR: $regex does not adhere to format - must contain [12]" unless $regex=~/\[12\]/; 

# make sample paths
my $pe1 = "";
my $pe2 = "";
if ($input ne ""){
	my $regex_1 = $regex; 
	$regex_1 =~ s/\[12\]/1/g;
	$pe1 = "$input$sample$regex_1";
	 
	my $regex_2 = $regex; 
	$regex_2 =~ s/\[12\]/2/g;
	$pe2 = "$input$sample$regex_2";
}else{
	$pe1 = $R1;
	$pe2 = $R2;
}

# check input files exist
unless ( ( -f "$pe1" ) && ( -f "$pe2" ) ){
	print LOG " - input files do not exist\n";
	close LOG;
	die " - ERROR: input files do not exist\n";
}

# working file variables
my $working_1 = $pe1;
my $working_2 = $pe2;

# copy files to output directory and set as working files
if ( $pe1 =~ /\.gz$/ ){
	system("cp $pe1 $output_dir/$sample\_1.fq.gz");
	$working_1 = "$output_dir/$sample\_1.fq.gz";
}else{
	system("cp $pe1 $output_dir/$sample\_1.fq");
	$working_1 = "$output_dir/$sample\_1.fq";
}
if ( $pe2 =~ /\.gz$/ ){
	system("cp $pe2 $output_dir/$sample\_2.fq.gz");
	$working_2 = "$output_dir/$sample\_2.fq.gz";
}else{
	system("cp $pe2 $output_dir/$sample\_2.fq");
	$working_2 = "$output_dir/$sample\_2.fq";
}

# down-sample if coverage is > $threshold

# unzip if necessary
if ( $working_1 =~ /\.gz$/ ){
	system("gunzip -kc $working_1 > $output_dir/$sample\_1.fq");
	unlink($working_1);
	$working_1 = "$output_dir/$sample\_1.fq";	
}
if ( $working_2 =~ /\.gz$/ ){
	system("gunzip -kc $working_2 > $output_dir/$sample\_2.fq");
	unlink($working_2);
	$working_2 = "$output_dir/$sample\_2.fq";	
}	

# run subsample reads
my $tmp_gs = $genome_size;
$tmp_gs = 10000000 if $genome_size == 0;
my $sub_reads1 = `$script_path/Subsample_reads $working_1 $working_2 $tmp_gs $upper`;
print LOG "initial coverage/subsampling:\n$sub_reads1\n";
die " - ERROR: error in subsample_reads\n" if $?;  

# check coverage
$sub_reads1 =~ /Estimated coverage = (\S+)/;
my $cov = $1;

# check if file has been subsampled
if( ( -f "$working_1.reduced") && ( -f "$working_2.reduced") ){
	
	# [optional] remove original files
	system("rm $working_1 $working_2") unless $keep_intermediate == 1;

	# set new working files
	$working_1 = "$working_1.reduced";
	$working_2 = "$working_2.reduced";

}

# [default] run trimmomatic
unless ($no_trim == 1){
	
	# trim log files
	my $trim_log = "$output_dir/$sample.trim_log";
	
	# set trimmomatic file paths
	my $paired_1 = "$output_dir/$sample\_1.paired.fq";
	my $paired_2 = "$output_dir/$sample\_2.paired.fq";
	my $unpaired_1 = "$output_dir/$sample\_1.unpaired.fq";
	my $unpaired_2 = "$output_dir/$sample\_2.unpaired.fq";

	# set trimmomatic options
	my @trim_opts = split(/\s+/, $trim_opts); 
	push( @trim_opts, "TOPHRED33");
	my $trim_opts_in = join(" ", @trim_opts);
	
	# run trimmomatic
	my $trim_command = "trimmomatic PE -threads $cores $working_1 $working_2 $paired_1 $unpaired_1 $paired_2 $unpaired_2 $trim_opts_in 2>$trim_log";
	
	print LOG "trimmomatic:\ncommand - $trim_command\n";
	system("$trim_command");
	my $out_code = $?;
	
	# if trimmomatic failed try -phred33 encoding
	if ($out_code == 256){
		$trim_command = "trimmomatic PE -phred33 -threads $cores $working_1 $working_2 $paired_1 $unpaired_1 $paired_2 $unpaired_2 $trim_opts_in 2>$trim_log";
		system("$trim_command");
		$out_code = $?;		
	}
	
	# log results appropriately
	close LOG;
	system( "cat $log_file $trim_log > $output_dir/$sample.txt && mv $output_dir/$sample.txt $log_file" );
	
	# [optional] remove log file
	system("rm $trim_log") unless $keep_intermediate == 1;
	
	# [optional] remove pre-trimmed files
	unlink ($working_1) unless $keep_intermediate == 1;
	unlink ($working_2) unless $keep_intermediate == 1;
		
	# catch failed trimmomatic run
	if ( $out_code == 256 ){
		cleanup();
		die " - ERROR: trimmomatic did not complete, see $log_file\n";
	}
	
	# [optional] remove unpaired files
	system("rm $unpaired_1 $unpaired_2") if $keep_intermediate == 0;
 	
	# reopen log file 
	open LOG, ">>$log_file" or die " - ERROR: could not open log file($log_file)";
	
	# set working files
	$working_1 = $paired_1;
	$working_2 = $paired_2;

}

# [optional] down-sample if coverage is > $threshold - alternatively just count reads and length
my $temp_genome_size = $genome_size;
$temp_genome_size = "1000000000" if $temp_genome_size == 0 ;  # set dummy genome size if no ref passed

# run subsample reads
my $sub_reads2 = `$script_path/Subsample_reads $working_1 $working_2 $temp_genome_size $coverage`;
print LOG "\npost-trim coverage/subsampling:\n$sub_reads2\n";
die " - ERROR: error in subsample_reads\n" if $?;   

# check coverage
$sub_reads2 =~ /Estimated coverage = (\S+)/;
$cov = $1;

# check for reduced files	
if( ( -f "$working_1.reduced") && ( -f "$working_2.reduced") ){

	# [optional] remove original files
	system("rm $working_1 $working_2") unless $keep_intermediate == 1;
	
	# set new working files
	$working_1 = "$working_1.reduced";
	$working_2 = "$working_2.reduced";
	
}

# run musket for kmer spectrum read correction

if ($no_musket == 0){

	# estimate number of K-mers for each K based on genome size
	$musket_ks =~ s/\s+//g;
	my @ks = split(/,/, $musket_ks);
	my @musket_ks = ();
	for my $k (@ks){
		my $est = "";
		if ($genome_size == 0){
			$est = 536870912;
		}else{
			my $est_ks = ( $genome_size - $k ) + 1;
			$est = int($est_ks + (0.1 * $est_ks)); # N=L-k+1 + error rate (0.1)
		}
	
		push(@musket_ks, "-k $k $est");
	}
	my $ks_in = join(" ", @musket_ks);
	$ks_in = $ks_in." -multik 1" if @ks > 1;

	# trim musket opts to not contain zlib
	$musket_opts =~ s/-zlib 1//g;

	# run musket
	my $musket_log = "$output_dir/musket.log";
	`$script_path/musket-1.1/musket -p $cores $musket_opts $ks_in -omulti $output_dir/musket -inorder $working_1 $working_2 > $musket_log 2> $musket_log`;
	die " - ERROR: musket failed - see $musket_log\n" if $?; 

	# make new files working files
	`mv $output_dir/musket.0 $working_1.musket`;
	`mv $output_dir/musket.1 $working_2.musket`;

	# [optional: remove previous files]
	system("rm $working_1 $working_2") unless $keep_intermediate == 1;

	# set new working files
	$working_1 = "$working_1.musket";
	$working_2 = "$working_2.musket";  

}

# rename final file
system ("cp $working_1 $output_dir/$sample\_1.fastq");
system ("cp $working_2 $output_dir/$sample\_2.fastq");

# [optional: remove previous files]
system("rm $working_1 $working_2") unless $keep_intermediate == 1;

# set working files
$working_1 = "$output_dir/$sample\_1.fastq";
$working_2 = "$output_dir/$sample\_2.fastq";

# run fastqc
my $fastqc_command = "fastqc --outdir=$output_dir --dir=$output_dir --quiet $working_1 $working_2"; 
print LOG "fastqc:\n$fastqc_command\n";
my $fastqc_log = `$fastqc_command`; 
if ($?){
	print LOG "$fastqc_log\n - fastqc failed\n";
	die " - ERROR: fastqc did not complete\n" if $?;
}else{
	print LOG "$fastqc_log\n - fastqc completed";
}

# gzip final
my $gz_err_1 = system("gzip -f $working_1 2>&1");
my $gz_err_2 = system("gzip -f $working_2 2>&1");

if ($gz_err_1 || $gz_err_2){
	print LOG "\n- ERROR: could not compress reads - see below\n$gz_err_1\n$gz_err_2\n";
	die "\n- ERROR: could not compress reads - see log\n";
}

# feedback
print LOG "\n - qc-pipe completed successfully\n";
close LOG;

# summarise downsampling and trimming
my $ref_l = "NA";
$ref_l = $genome_size unless ( $genome_size == "1000000000" ); # unnecessary

my $no_reads_pre = "NA";
my $av_readl_pre = "NA";
my $cov_pre = "NA";

my $no_read_pre_trim = "NA";

my $no_reads_post = "NA";
my $av_readl_post = "NA";
my $cov_post = "NA";

# parse log file
my $pre_post_switch = 0;
open LOG, $log_file or die " - ERROR: could not open log file - $log_file\n";
while (<LOG>){

	my $line = $_;
	chomp $line;
	
	if( /post-trim coverage\/subsampling:/ ){
		$pre_post_switch = 1;
	}if(/Input Read Pairs: (\d+)/){
		$no_read_pre_trim = $1;	
	}elsif( $pre_post_switch == 0){
	
		if (/Number of reads = (\S+)/){
			$no_reads_pre = $1;
		}elsif(/Average read length = (\S+)/){
			$av_readl_pre = $1;
		}elsif(/Estimated coverage = (\S+)/){
			$cov_pre = $1; # only include if ref provided
		}
	
	}elsif( $pre_post_switch == 1){
	
		if (/Number of reads = (\S+)/){
			$no_reads_post = $1;
		}elsif(/Average read length = (\S+)/){
			$av_readl_post = $1;
		}elsif(/Estimated coverage = (\S+)/){
			$cov_post = $1; # only include if ref provided
		}
	
	}


}close LOG;

# print summary line to file.
my $summary_file = "$output_dir/$sample.summary.tab";
open SUMMARY, ">$summary_file" or die " - ERROR: could not make summary file\n";
print SUMMARY "#reference_length\torg_no._reads\torg_read_l\torg_cov\tno_reads_pre_trim\tno_reads_post_trim\tav_read_l_post_trim\tcov_post_trim\n";
print SUMMARY "$ref_l\t$no_reads_pre\t$av_readl_pre\t$cov_pre\t$no_read_pre_trim\t$no_reads_post\t$av_readl_post\t$cov_post\n";
close SUMMARY;

# cleanup temp files
cleanup();

# subfunctions
sub cleanup{

	if ($keep_intermediate == 0){
	
		my $sample_root = "$output_dir/$sample";
	
		unlink("$sample_root\_1.fq");
		unlink("$sample_root\_2.fq");
		unlink("$sample_root\_1.fq.reduced");
		unlink("$sample_root\_2.fq.reduced");
		unlink("$sample_root\_1.paired.fq");
		unlink("$sample_root\_2.paired.fq");
		unlink("$sample_root\_1.unpaired.fq");
		unlink("$sample_root\_2.unpaired.fq");
		unlink("$sample_root\_1.paired.fq.reduced");
		unlink("$sample_root\_2.paired.fq.reduced");
		unlink("$sample_root.trim_log");
	}
}

exit
