#!/usr/bin/env perl

use strict;
use warnings;
use Getopt::Long qw(GetOptions);
use Pod::Usage;
use Cwd 'abs_path';
use File::Basename;

# parse file list, check for file presence and check for existing files in output folder 

=head1  SYNOPSIS

 parse_list.pl [--input /path/to/input_list] [--ref /path/to/reference_genome] [--output /path/to/output_directory] [opt_args]
 
 Input options:
 --input 	input sample list [required]
 --ref		reference genome for mapping in fasta/gbk format [optional]
 --output	path to output directory [required]
 --runid	prefix for output files [required]
  
 Options:
 --options	options to pass to wgs-pipe for all samples [optional]
 --reads	input files are local reads [optional]

 General options:
 --force	force assembly/mapping over previous files [default: off]
 -h|--help	usage information
 
=cut

# path to executing script
my $script_path = abs_path(dirname($0));

# command line options
my $input = '';
my $output = '';

my $local = 0;
my $contigs = 0;
my $remote = 0;

my $general_options = "";
my $readqc_options = "";
my $unitig_options = "";
my $mapping_options = "";
my $assembly_options = "";
my $annotation_options = "";
my $assemblyqc_options = "";

my $read_qc_off = 0;
my $unitig_off = 0;
my $mapping_off = 0;
my $annotation_off = 0;
my $assembly_off = 0;
my $assembly_qc_off = 0;

my $ref = '';
my $runid = '';

my $force = 0;
my $help = 0;

GetOptions(

	'input=s' => \$input,
	'output=s' 	=> \$output,
	'runid=s' => \$runid,
	
	'ref=s' => \$ref,
	
	'reads' => \$local,
	'remote' => \$remote,
	'contigs' => \$contigs,
	
	'general-options=s' => \$general_options,
	'readqc-options=s' => \$readqc_options,
	'unitig-options=s' => \$unitig_options,
	'mapping-options=s' => \$mapping_options,
	'assembly-options=s' => \$assembly_options,
	'annotation-options=s' => \$annotation_options,
	'assemblyqc-options=s' => \$assemblyqc_options,
	
	'no-readqc' => \$read_qc_off,
	'no-unitigs' => \$unitig_off,
	'no-mapping' => \$mapping_off,
	'no-annotation' => \$annotation_off,
	'no-assembly' => \$assembly_off,
	'no-assemblyqc' => \$assembly_qc_off,
	
	'force' => \$force,
	'help|?' => \$help,
				
) or pod2usage(1);
pod2usage(1) if $help;

# check inputs
pod2usage( {-message => q{output path is a required arguement}, -exitval => 1, -verbose => 1 } ) if $output eq ''; 
pod2usage( {-message => q{input file is a required arguement}, -exitval => 1, -verbose => 1 } ) if $input eq ''; 
pod2usage( {-message => q{runid is a required arguement}, -exitval => 1, -verbose => 1 } ) if $runid eq ''; 

# check output directory exists
unless ( -e $output ){
	die " - ERROR: could not make output directory ($output)\n" unless mkdir($output); 
} 
$output = abs_path($output);
# check log directory exists
my $logs = "$output/logs";
unless ( -e $logs ){
	die " - ERROR: could not make output directory ($logs)\n" unless mkdir($logs); 
} 
$logs = abs_path($logs);

# set general options
my $parse_opts = "";
$parse_opts = $parse_opts." --force" if $force == 1;

# module toggle
$parse_opts = $parse_opts." --no-readqc" if $read_qc_off == 1;
$parse_opts = $parse_opts." --no-unitigs" if $unitig_off == 1;
$parse_opts = $parse_opts." --no-mapping" if $mapping_off == 1;
$parse_opts = $parse_opts." --no-assembly" if $assembly_off == 1;
$parse_opts = $parse_opts." --no-annotation" if $annotation_off == 1;
$parse_opts = $parse_opts." --no-assemblyqc" if $assembly_qc_off == 1;

# create output file containing commands for wgs-pipe
open OUT , ">$output/$runid.commands.txt" or die " - ERROR: could not open commands file\n";

# parse list file 
open IN, $input or die " - ERROR: could not open input list file\n"; 
while(<IN>){
	
	my $line = $_;
	chomp $line;
	
	my @var = split("\t", $line, -1);
	
	# ignore blank lines
	if (/^\S+/){
	
		# sample name
		my $name = $var[0];
		
		# set variables 
		my $warnings = 0;
		my $ref_file = $ref;
		my @wgs_options = ();
		my $input_path = "";
		
		# input options
		if ($remote){
			$input_path = "--remote $var[1]"; 
		}
		if ($contigs){
			$input_path = "--contigs $var[1]"; 
		}
		if ($local){
			$input_path = "--R1 $var[1] --R2 $var[2]"; 
		}
		
		# module options
		my $read_qc = "";
		my $unitig = "";
		my $mapping = "";
		my $assembly = "";
		my $annotation = "";
		my $assembly_qc = "";
		
		# parse additional options
		my $options = $var[2];
		$options = $var[3] if $local == 1;
	
		# split options
		if ( defined($options) ){
			
			for my $option ( split(/;/, $options) ){
				
				if ( $option =~ /^reference=\"(.+)\"/ ){			
					$ref_file = $1;					
				}
				elsif ( $option =~ /^readqc=\"(.+)\"/ ){
					$read_qc = $read_qc.$1;
				}
				elsif ( $option =~ /^unitig=\"(.+)\"/ ){
					$unitig = $unitig.$1;
				}
				elsif ( $option =~ /^mapping=\"(.+)\"/ ){
					$mapping = $mapping.$1;
				}
				elsif ( $option =~ /^assembly=\"(.+)\"/ ){
					$assembly = $assembly.$1;
				}
				elsif ( $option =~ /^annotation=\"(.+)\"/ ){
					$annotation = $annotation.$1;
				}
				elsif ( $option =~ /^assemblyqc=\"(.+)\"/ ){
					$assembly = $assembly.$1;
				}			
			}
		}
		
		# set ref abs path
		if ($ref_file ne "" ){
			if ( !(-f $ref_file) ){
				print "$name\treference_file ($ref_file) does not exist\n";
				$warnings++;
			}else{
				$ref_file = abs_path($ref_file);
				push(@wgs_options , "--ref $ref_file" );
			}
		}
		
		# set reference name 
		my $ref_name = basename(abs_path($ref_file));
	 	$ref_name =~ s/(.fna$)||(.fa$)||(.fas$)||(.fasta$)||(.gb$)||(.gbk$)//g;
		
		# set path
		my $sample_dir = abs_path("$output/$name");
		
		# general module options
		push( @wgs_options , "--readqc-options \"$readqc_options\"" ) if $readqc_options ne "";
		push( @wgs_options , "--unitig-options \"$unitig_options\"" ) if $unitig_options ne "";
		push( @wgs_options , "--mapping-options \"$mapping_options\"" ) if $mapping_options ne "";
		push( @wgs_options , "--assembly-options \"$assembly_options\"" ) if $assembly_options ne "";
		push( @wgs_options , "--annotation-options \"$annotation_options\"" ) if $annotation_options ne "";
		push( @wgs_options , "--assemblyqc-options \"$assemblyqc_options\"" ) if $assemblyqc_options ne "";

		# individual module options
		push( @wgs_options , "--readqc-options \"$read_qc\"" ) if $read_qc ne "";
		push( @wgs_options , "--mapping-options \"$mapping\"" ) if $mapping ne "";
		push( @wgs_options , "--assembly-options \"$assembly\"" ) if $assembly ne "";
		push( @wgs_options , "--annotation-options \"$annotation\"" ) if $annotation ne "";
		push( @wgs_options , "--assemblyqc-options \"$assembly_qc\"" ) if $assembly_qc ne "";

		# print command to file if there were no warnings
		if ($warnings == 0){
			my $print_opts = join(" ", @wgs_options);
			print OUT "perl $script_path/wgs-pipe.pl --sample-id $name $input_path --output $sample_dir $print_opts $parse_opts $general_options > $logs/$name.$runid.log\n";
		}
		
	}	
}close IN;

