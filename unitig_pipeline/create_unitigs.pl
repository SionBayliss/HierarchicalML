#!/usr/bin/env perl

use strict;
use warnings;
use Getopt::Long qw(GetOptions);
use Pod::Usage;
use File::Basename;
use Cwd;
use Cwd 'abs_path';
use File::Copy;

# create unitigs and get k-mer count using bcalm2

=head1  SYNOPSIS

 create_unitigs -i /path/to/input_file -o /path/to/output_directory/
 
 Input/Output:
 --R1		path to read pair 1 [required with --R2]
 --R2		path to read pair 2 [required with --R1]
 --input	path to input fasta [required unless --R1 and --R2]
 --output	output directory [required]
 --name		sample name [required]
 		
 Bcalm Options:
 --kmer		K-mer size for bcalm [default: 31]
 --bcalm-opts	BCALM options [default: "-histo 
 		--all-abundance-counts -abundance-min 2"]
 
 General options:
 --force	force replace files [default: off]
 -z		keep intermediate files [default: off]
 --threads	number of threads [default: 1]
 --help		usage information
 
=cut

# script path
my $script_path = abs_path(dirname($0));

# command line options
my $R1 = '';
my $R2 = '';
my $input = '';
my $sample = '';
my $output_dir = '';

my $bcalm_opts = "-kmer-size 31 -abundance-min 2";

my $keep_intermediate = 0;
my $test = 0;
my $cores = 1;
my $no_trim = 0;
my $force = 0;
my $help = 0;

GetOptions(

	'help|?' 	=> \$help,
	
	'R1=s' => \$R1,
	'R2=s' => \$R2,
	'input=s' 	=> \$input,
	'name=s' 	=> \$sample,
	'output=s'	=> \$output_dir,
	
	'bcalm-opts=s'	=> \$bcalm_opts,
	
	'threads=i' => \$cores,
	'test' => \$test,
	'force' => \$force,
	'z' => \$keep_intermediate,
			
) or pod2usage(1);
pod2usage(1) if $help;

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
$output_dir = abs_path($output_dir);

# make log file
my $log_file = "$output_dir/log.file";
open LOG, ">$log_file" or die " - ERROR: could not make log file\n";

# check input files exist
unless ( (( -f "$R1" ) && ( -f "$R2" )) || ( -f $input) ){
	print LOG " - input files do not exist\n";
	close LOG;
	die " - ERROR: input files do not exist\n";
}

# make list for bcalm
my $list_file = "$output_dir/list.file";
open LIST, ">$list_file" or die " - ERROR: could not make list file\n";
if ( ($input ne '') || ( ($R1 eq "") && ($R2 eq "") ) ){
	print LIST abs_path($input)."\n";
}else{
	print LIST abs_path($R1)."\n".abs_path($R2)."\n";
}
close LIST;

# run bcalm
if( (!(-f "$output_dir/$sample.unitigs.fa") && !(-f "$output_dir/$sample.unitigs.fa.gz")) || ($force == 1) ){
	
	# get current dir 
	my $curdir = cwd;
	
	# make temp dir for hd5 file
	my $temp_dir = "$output_dir/temp";
	mkdir("$temp_dir");
	chdir("$temp_dir");
	
	# run bcalm
	my $bcalm_file = "$output_dir/bcalm_log.txt";
	
	# check default options set
	$bcalm_opts =~ s/\-nb\-cores \d+//g; # cores controlled using speciic option
	$bcalm_opts = $bcalm_opts." -kmer-size 31" unless ($bcalm_opts =~ /-kmer-size \d+/); # kmer must be set
	
	# note: h5 file only created if -out option not set - so temp dir created and curruent directory changed
	my $bcalm_command = "bcalm $bcalm_opts -nb-cores $cores -in $list_file -out-dir $output_dir -out-tmp $output_dir -histo --all-abundance-counts > $bcalm_file 2> $bcalm_file";
	print LOG "command: $bcalm_command\n";
	system("$bcalm_command");
	die " - ERROR: BCALM failed: see $bcalm_file for details\n" if $?;
	die " - ERROR: BCALM failed: see $bcalm_file for details\n" if !(-f "$temp_dir/list.unitigs.fa");
	
	# move unitigs file and remove temp dir
	move("$temp_dir/list.unitigs.fa", "$output_dir/$sample.unitigs.fa");
	
	# move back to prev dir
	chdir($curdir);

	# remove temp dir
	rmdir $temp_dir;
	
}


# save histogram
if( (!(-f "$output_dir/$sample.histogram.tab") && !(-f "$output_dir/$sample.histogram.tab.gz")) || ($force == 1)  ){

	my $hist_h5 = "$output_dir/hist.temp";
	system("h5dump -d /histogram/histogram/ $output_dir/list.h5 >$hist_h5 2>/dev/null");

	open H5, $hist_h5 or die " - ERROR: could not open $hist_h5\n";
	open HIST, ">$output_dir/$sample.histogram.tab" or die " - ERROR: could not open $output_dir/$sample.histogram.tab\n";
	my $store = 0;
	my @otemp = ();
	while(<H5>){

		if ($store==1){
			if(/^\s+(\d+)\,\n/){
				push(@otemp, $1);
			}elsif(/^\s+(\d+)\n/){
				push(@otemp, $1);
				print HIST join("\t", @otemp)."\n";
				@otemp = ();
			}
		}elsif(/DATA \{/){
			$store=1;
		}
	
	}
	close H5;
	close HIST;
	
}

# summary stats

# summary vars
my $unitig_count = 0;
my $base_count = 0;
my $kmer_count = 0;
my $singletons = 0;

# open unitig file
my $unitig_file = "$output_dir/$sample.unitigs.fa";
if ( !(-f $unitig_file) && (-f "$unitig_file.gz") ){
	system("gunzip $unitig_file.gz");
}
open UNITIGS, $unitig_file or die " - ERROR: could not open $unitig_file file\n";
while(<UNITIGS>){
	my $line =$_;
	chomp $line;
	
	if (/^>/){
		$unitig_count++;
	}else{
		$base_count+=length($line);
	}
}close UNITIGS;

# open histogram file
my $hist_file = "$output_dir/$sample.histogram.tab";
if ( !(-f $hist_file) && (-f "$hist_file.gz") ){
	system("gunzip $hist_file.gz");
}
open HIST2, $hist_file or die " - ERROR: could not open $hist_file file\n";
while(<HIST2>){
	my $line =$_;
	chomp $line;
	
	if (/^(\d+)\t(\d+)/){
		if($1 == 1){
			$singletons = $2;
		}
		$kmer_count+=$2;
	}
	
}close HIST2;

my $perc_single = ($singletons/$kmer_count)*100;
my $sum_line = sprintf("$unitig_count\t$base_count\t$kmer_count\t$singletons\t%.2f\n", $perc_single);
print LOG "\nunitig_count\tbase_count\tkmer_count\tsingletons\t\%single\n";
print LOG $sum_line;

# gzip output files
if( -f "$output_dir/$sample.unitigs.fa" ){
	system("gzip -f $output_dir/$sample.unitigs.fa");
}
if( -f "$output_dir/$sample.histogram.tab" ){
	system("gzip -f $output_dir/$sample.histogram.tab");
}

# clean up
unless($keep_intermediate){
	unlink glob "$output_dir/*glue*";
	unlink "$output_dir/bcalm_log.txt";
	unlink "$output_dir/list.file";
	unlink "$output_dir/hist.temp";
	unlink "$output_dir/list.h5";
}

close LOG;
