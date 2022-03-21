#!/usr/bin/env perl

use strict;
use warnings;
use Getopt::Long qw(GetOptions);
use Pod::Usage;
use Cwd 'abs_path';
use File::Basename;

# collect summary files for a list of samples processed by wgs-pipe.

=head1  SYNOPSIS

 summarise_set.pl [--input /path/to/input_list] [--output /path/to/output_directory] [opt_args]
 
 Input options:
 --input 	input sample list [required]
 --results 	output directory from wgs-pipe [required]
 --output	path to output directory [required]

 Output options: 
 --prefix	output file prefixes [optional]
 --summary-list comma seperated list of files to summarise 
 		[default: snps,assembly,mapping,read-qc,assembly-qc]
 --keep-aln	keep alignment files for all contigs [default off] 
 --unitig-hist	create concatenated version of kmer histograms 
 		[default: off]
 
 General options:
 --force	force overwrite previous files [default: off]
 -h|--help	usage information
 
=cut

# path to executing script
#my $script_path = abs_path(dirname($0));

# command line options
my $input = '';
my $input_dir = '';
my $output = '';

my $unitig_hist = 0;

my $prefix = '';
my $list = "snps,unitigs,assembly,mapping,read-qc,assembly-qc";
my $keep_contig_aln = 0;
my $force = 0;
my $help = 0;

# generate accepted values for list
my @accept = split(/,/, $list);
my %accepted_values = ();
for (@accept) { $accepted_values{$_} = 1 };

GetOptions(

	'input=s' => \$input,
	'results=s' => \$input_dir, 
	'output=s' 	=> \$output,
	
	'prefix=s' => \$prefix,
	'summary-list=s' => \$list, 
      'keep-aln' => \$keep_contig_aln,
      'unitig-hist' => \$unitig_hist,

	'force' => \$force,
	'help|?' => \$help,
				
) or pod2usage(1);
pod2usage(1) if $help;

# check inputs
pod2usage( {-message => q{output path is a required arguement}, -exitval => 1, -verbose => 1 } ) if $output eq ''; 
pod2usage( {-message => q{input file is a required arguement}, -exitval => 1, -verbose => 1 } ) if $input eq ''; 
pod2usage( {-message => q{input directory is a required arguement}, -exitval => 1, -verbose => 1 } ) if $input_dir eq ''; 

# check output directory exists
unless ( -e $output ){
	die " - ERROR: could not make output directory ($output)\n" unless mkdir($output); 
} 

# make paths absolute
$output = abs_path($output);
$input_dir = abs_path($input_dir);

# check summary list
my @list  = split(/,/, $list);
for (@list){ die " - $_ is not an accepted input for --summary-list. Use snps,assembly,mapping,read-qc,assembly-qc" unless $accepted_values{$_} ;}
my %summarize = ();
for (@list) { $summarize{$_} = 1 };

# check for compatible options
if ( $unitig_hist && !$accepted_values{"unitigs"} ){
	print " - cannot create unitig histogram without unitig summary\n";
}

# make missing file variable 
my %missing = ();

# parse sample list - first column is sample name
my %sample_hash = ();
open IN, $input or die " - ERROR: could not open $input\n";
while (<IN>){

	my $line = $_;
	chomp $line;
	
	my @line  = split(/\t/, $line, -1);
	
	my $st = $line[0];
	
	if ($line =~/^\S+/) {
		if ( -e "$input_dir/$st/" ){
			$sample_hash{$st} = 1;
		}else{	
			
			# feedback
			print " - WARNING: sample $st does not match any output directories\n";
			
			# add to missing list	
			for (@list) { $missing{$_}{$st} = 1 };
		}
	}
}
my @sample_list = keys(%sample_hash);

# collect all appropriate summary file specified by $list.
unless (!$summarize{"read-qc"}){

	# make output summary file.
	my $qc_summary = "$output/read_qc_summary.tsv";
	open QCSUM, ">$qc_summary" or die " - ERROR: could not open read QC summary file ($qc_summary)\n";
	
	# add headers 
	print QCSUM "#sample_id\treference_length(bp)	org_no._reads	org_read_l	org_cov	no_reads_pre_trim	no_reads_post_trim	av_read_l_post_trim	cov_post_trim\n";
	
	for my $s (@sample_list){
	
		my $qc_summary_file = "$input_dir/$s/read-qc/$s.summary.tab";
		
		# check summary files exist.
		if ( -f "$qc_summary_file" ){
			
			open QC_TEMP, "$qc_summary_file" or die " - ERROR: could not open $qc_summary_file for reading\n";			
			while (<QC_TEMP>){
				
				print QCSUM "$s\t$_" if $_ !~ /^#/;
								
			}close QC_TEMP;				
		}
		# if file missing add NAs
		else{
			print QCSUM "$s\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\n";
		}	
	}
	close QCSUM;
}


# collect all appropriate summary file specified by $list.
unless (!$summarize{"unitigs"}){

	# make output summary file.
	my $uni_summary = "$output/unitig_summary.tsv";
	open UNSUM, ">$uni_summary" or die " - ERROR: could not open unitig log file ($uni_summary)\n";
	
	# add headers 
	print UNSUM "#sample_id\tunitig_count\tbase_count\tkmer_count\tsingletons\t\%single\n";
	
	# open histogram file [optional]
	my $UH;
	if ( $unitig_hist ) {
		open $UH, ">$output/unitig_histogram.tsv" or die " - ERROR: could not open unitig histogram file\n";
	}
	
	for my $s (@sample_list){
	
		my $summary_file = "$input_dir/$s/unitigs/log.file";
		
		# check summary files exist.
		my $print = 0;
		my $printed = 0;
		if ( -f "$summary_file" ){
			
			open UN_TEMP, "$summary_file" or die " - ERROR: could not open $summary_file for reading\n";			
			while (<UN_TEMP>){
				
				if (/^unitig_count/){
					$print = 1;
				}elsif($print == 1){
					print UNSUM "$s\t$_" if $_ !~ /^#/;
					$print = 0;
					$printed = 1;
				}
				
								
			}close UN_TEMP;				
		}
		
		# if file missing add NAs
		unless($printed){
			print UNSUM "$s\tNA\tNA\tNA\tNA\tNA\n";
		}	
		
		# summarise histograms [optional]
		if ( $unitig_hist ) {
			
			my $hist_file = "$input_dir/$s/unitigs/$s.histogram.tab.gz";
			my $temp_hist_file = "$input_dir/$s/unitigs/$s.histogram.tab.gz.temp";
			
			if ( -f $hist_file){
			
				# decompress
				system("gunzip -c $hist_file > $temp_hist_file");
				
				open UH_TEMP, "$temp_hist_file" or die " - could not open histogram file - $temp_hist_file";
				my @out_line = ($s);
				while(<UH_TEMP>){
						
						my $line = $_;
						chomp $line;
						
						my @vars = split(/\t/, $line, -1);
						
						# sanity check 
						die " - malformed line in $temp_hist_file - $line\n" if (@vars != 2);
						
						# store in variable 
						push(@out_line, $vars[1]);
				}
				close UH_TEMP;
				
				# print to file	
				my $oline = join("\t", @out_line);
				print $UH "$oline\n";
				
				# remove temp file
				unlink("$temp_hist_file");
			}
			
		}
	}
	close UNSUM;
	
}

exit
