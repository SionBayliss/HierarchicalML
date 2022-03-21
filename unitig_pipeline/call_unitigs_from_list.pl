#!/usr/bin/env perl

use strict;
use warnings;
use Getopt::Long qw(GetOptions);
use Pod::Usage;
use File::Basename;
use Cwd;
use Cwd 'abs_path';
use File::Copy;

# call presence/absence of unitigs in collection using unitig-caller

=head1  SYNOPSIS

 call_unitigs_from_list -i /path/to/input_file -o /path/to/output_directory/
 
 Input/Output:
 --results 	path to results directory [required]
 --list 	path to sample list [required]
 --output	path to output directory [required]
 --query		path to query unitig fasta [optional,
 		default: graph from input collection]
 
 General options:
 --threads	number of threads [default: 1]
 --help		usage information
 
=cut

# script path
my $script_path = abs_path(dirname($0));

# command line options
my $results = "";
my $list = "";
my $output_dir = "";
my $prev_unitigs = "";

my $cores = 1;
my $help = 0;

GetOptions(

	'help|?' 	=> \$help,
	
	'output=s'	=> \$output_dir,
	'list=s'  => \$list,
	'results=s' => \$results,
	'query=s' => \$prev_unitigs,
	
	'threads=i' => \$cores,
			
) or pod2usage(1);
pod2usage(1) if $help;

# check mandatory inputs
pod2usage( {-message => q{results directory is a required argument}, -exitval => 1, -verbose => 1 } ) if $results eq ''; 
pod2usage( {-message => q{output directory is a required argument}, -exitval => 1, -verbose => 1 } ) if $output_dir eq ''; 
pod2usage( {-message => q{list directory is a required argument}, -exitval => 1, -verbose => 1 } ) if $list eq ''; 

# check output directory exists
unless( -e $output_dir ){ die " - ERROR: output directory does not exist and cannot be created\n" unless mkdir($output_dir)};
$output_dir = abs_path($output_dir);

# make log file
#my $log_file = "$output_dir/log.file";
#open LOG, ">$log_file" or die " - ERROR: could not make log file\n";

# parse list 
my @samples = ();
open LIST, $list or die " - ERROR: could not open list file\n";
while(<LIST>){

	if (/^(\S+)/){
		push(@samples, $1);
	}
	
}close LIST;

# feedback 
print " - ".@samples." samples in list\n";

# check unitig files exist and unzip
my @temp_samples = ();
my @temp_paths = ();
my $sample_count = ();
for my $s (@samples){

	my $file_path = sprintf("%s/%s/unitigs/%s.unitigs.fa", $results, $s, $s);
	$file_path =~ s/\/\//\//g;
 	my $gz_path =  "$file_path.gz";
 	 
	if( -f $gz_path ){
	
		system("gunzip -c $gz_path > $file_path");
		
		if ($?){
			print " - FILE ERROR: could not unzip $s - $gz_path\n";
		}else{
			push(@temp_samples, $s);
			push(@temp_paths, $file_path);
		}
		
	}else{
		print " - FILE MISSING: $s - $gz_path\n";
	}
	
}


# check for presence of previous unitigs graph
my $out_unitigs = "$output_dir/unitigs_unitigs.fasta";
my $out_bfg = "$output_dir/unitigs.bfg_colours";
my $out_gfa = "$output_dir/unitigs.gfa";

my $run = 1;
$run = 0 if ( (-f $out_unitigs) && (-f $out_bfg) && (-f $out_gfa));

if ($run == 1){

	# create refs file 
	open REFS, ">refs.txt" or die " ERROR: could not open refs.txt\n";
	for my $i (@temp_paths){
		print REFS "$i\n";
	} close REFS;

	# invoke unitig caller
	print " - building unitigs for input collection\n";
	print ("$output_dir/unitigs\n");
	my $ub_command = "unitig-caller --build --refs refs.txt --threads $cores --output $output_dir/unitigs";
	my $ub_err = system($ub_command);
	die " - ERROR: unitig-caller build failed\n" if $?;

}

# compare presence/absence of newly- or previously-generated unitig list against the unitig graph of the current input collection
my $query_unitigs = $out_unitigs;
$query_unitigs = $prev_unitigs if ($prev_unitigs ne "");

print " - calling presence/absence of unitigs per sample\n";
my $uq_command = "unitig-caller --threads $cores --query --graph-prefix $output_dir/unitigs --unitigs $query_unitigs --output $output_dir/counts";
my $uq_err = system($uq_command);
die " - ERROR: unitig-caller query failed\n" if $?;

# clean up 
for my $i (@temp_paths){
	unlink($i);
}

print " - complete\n";
