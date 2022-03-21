#!/usr/bin/env perl

use strict;
use warnings;
use Getopt::Long qw(GetOptions);
use Pod::Usage;
use Cwd 'abs_path';
use File::Basename;

# Retrieves fastq from the ENA based on sample number. 
# help = http://www.ebi.ac.uk/ena/browse/read-download#downloading_files_ftp

=head1  SYNOPSIS

 remote_download.pl [--name ""] [--accession] [--output /path/to/output_dir]
 
 Input options: 
 --name		output file prefix [required]
 --accession	SRA/ENA accession number [required]
 
 Output options:
 --output 	path to output directory [required]
 
 General options: 
 --test 	run pipeline on test files
 -h|--help	usage information
 
 NOTE: Needs accession prefix check.
 
=cut

# path to executing script
my $script_path = abs_path(dirname($0));

# command line options
my $name = '';
my $sample = '';
my $output = '';

my $force = 0;
my $test = 0;
my $help = 0;

GetOptions(
		
	'output=s' 	=> \$output,
	
	'name=s' => \$name,
	'accession=s' => \$sample, 
	
	'force' => \$force,
	'test' => \$test,
	'help|?' => \$help,
			
) or pod2usage(1);
pod2usage(1) if $help;

# set test paths
if ($test == 1){
	$name = "rtest";
	$sample = "ERR351542";
	$output = "$script_path/../test/test";
}

# check inputs
pod2usage( {-message => q{ - ERROR: sample name (--name) required}, -exitval => 1, -verbose => 1 } ) if $name eq ''; 
pod2usage( {-message => q{ - ERROR: accession number (--accession) required}, -exitval => 1, -verbose => 1 } ) if $sample eq ''; 
pod2usage( {-message => q{ - ERROR: output path is a required arguement}, -exitval => 1, -verbose => 1 } ) if $output eq '';

# Check ER/SR/DR codes here
die " - ERROR: incorrect accession prefix, expected ER/SR/DR\n" unless ( ($sample =~ /^ER/) || ($sample =~ /^SR/) || ($sample =~ /^DR/));

# check output directory
unless ( -e $output ){
	die " - ERROR: could not make output directory ($output)\n" unless mkdir($output); 
}
$output = abs_path($output);

# find length of run accession number. 
my $length = length($sample);

# ENA/SRA downloads

# <dir1> is the first 6 letters and numbers of the run accession ( e.g. ERR000 for ERR000916 ),
my $dir1 = "";
my $dir2 = "";
my $file1 = "";
my $file2 = "";

# <dir2> does not exist if the run accession has six digits. For example, fastq files for run ERR000916 are in directory: ftp://ftp.sra.ebi.ac.uk/vol1/fastq/ERR000/ERR000916/.
if ($length == 9) { 

	$dir1 = substr( $sample, 0, 6 );
	$file1 = sprintf ('ftp://ftp.sra.ebi.ac.uk/vol1/fastq/%s/%s/%s_1.fastq.gz', $dir1, $sample, $sample);
	$file2 = sprintf ('ftp://ftp.sra.ebi.ac.uk/vol1/fastq/%s/%s/%s_2.fastq.gz', $dir1, $sample, $sample);

}

# if the run accession has seven digits then the <dir2> is 00 + the last digit of the run accession. For example, fastq files for run SRR1016916 are in directory: ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR101/006/SRR1016916/.
elsif ( $length == 10) {

	$dir1 = substr ( $sample, 0, 6 );
	my $end = substr( $sample, 9, 1);
	$dir2 = "00$end";

	$file1 = sprintf ('ftp://ftp.sra.ebi.ac.uk/vol1/fastq/%s/%s/%s/%s_1.fastq.gz', $dir1, $dir2, $sample, $sample);
	$file2 = sprintf ('ftp://ftp.sra.ebi.ac.uk/vol1/fastq/%s/%s/%s/%s_2.fastq.gz', $dir1, $dir2, $sample, $sample);

}
# if the run accession has eight digits then the <dir2> is 0 + the last two digits of the run accession. 
elsif ( $length == 11) {

	$dir1 = substr ( $sample, 0, 6 );
	my $end = substr( $sample, 9, 2);
	$dir2 = "0$end";

	$file1 = sprintf ('ftp://ftp.sra.ebi.ac.uk/vol1/fastq/%s/%s/%s/%s_1.fastq.gz', $dir1, $dir2, $sample, $sample);
	$file2 = sprintf ('ftp://ftp.sra.ebi.ac.uk/vol1/fastq/%s/%s/%s/%s_2.fastq.gz', $dir1, $dir2, $sample, $sample);

}

# if the run accession has nine digits then the <dir2> is the last three digits of the run accession. 
elsif ( $length == 12) {

	$dir1 = substr ( $sample, 0, 6 );
	my $end = substr( $sample, 9, 3);
	$dir2 = "$end";

	$file1 = sprintf ('ftp://ftp.sra.ebi.ac.uk/vol1/fastq/%s/%s/%s/%s_1.fastq.gz', $dir1, $dir2, $sample, $sample);
	$file2 = sprintf ('ftp://ftp.sra.ebi.ac.uk/vol1/fastq/%s/%s/%s/%s_2.fastq.gz', $dir1, $dir2, $sample, $sample);

}else{
	die " - ERROR: input did not adhere to expected nomenclature\n";
}

# outut file paths
my $ofile1 = "$output/$name\_1.fastq.gz";
my $ofile2 = "$output/$name\_2.fastq.gz";

# create command 
my $command1 = sprintf( "wget --passive-ftp %s -O %s", $file1, $ofile1); #--quiet
my $command2 = sprintf( "wget --passive-ftp %s -O %s", $file2, $ofile2);

# run wget
my $errors = 0;

`$command1`;
$errors++ if $?;

`$command2`;
$errors++ if $?;

# check files downloaded successfully
if ( ( -f "$ofile1" ) && ( -f "$ofile2" )  ){

	my $fsize1 = -s $ofile1;
	my $fsize2 = -s $ofile2;
	
	$errors++ if (( ( "$fsize1" == 0 ) && ( "$fsize2" == 0 )  ));

}else{
	
	$errors++;
		
}

# remove test files 
if ( $test == 1 ){
	
	# remove files 
	unlink "$ofile1" if -f "$ofile1";
	unlink "$ofile2" if -f "$ofile2";

	# remove test directory
	rmdir "$output";	
	
	# feedback
	print " - tests completed sucessfully\n" if $errors == 0;
	
}

# provide exit code if failed to download files
if ( $errors > 0 ){
	die " - ERROR: some files failed to download\n";
}

exit


