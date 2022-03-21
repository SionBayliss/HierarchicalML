#!/usr/bin/env perl

use strict;
use warnings;
use Getopt::Long qw(GetOptions);
use Pod::Usage;
use Cwd 'abs_path';
use File::Basename;
use POSIX;

# pipeline for wgs analysis of short read/assembly data. 

=head1  SYNOPSIS

 driver.pl [--contigs or --reads or --remote] [--output /path/to/output_dir]
 
 Input options: 
 --contigs		file containing contigs sample list [optional]
 --reads		file containing list of pe sample reads [optional]
 --remote		list of sample ids from ENA/SRA [optional]
 [required: at least one option --contigs, --reads or --remote]
 --reference 		reference fasta/gbk for mapping/alignment [optional]
 
 WGS-pipe options:
 --readqc-options	options to pass to read qc, applied to all samples [optional]
 --unitig-options	options to pass to bcalm, applied to all samples [optional]

  Module options:
 --no-readqc		switch off read qc module
 --no-unitigs		switch off unitig module
 
 Output options:
 --output 		path to output directory [required]
 
 General options: 
 --keep-intermediate		keep intermediate files [1 = read-qc, 2 = all]
 --keep-remote		keep remote files after downloading [default: off]
 --runid		run id for summary files [default: auto]
 --cpus			number of cpus to use [default: 1]
 --force 		force overwrite of pre-existing files
 --test 		run pipeline on test files
 -h|--help		usage information

=cut

# switch off buffering
$|=1;

# path to executing script
my $script_path = abs_path(dirname($0));

# command line options
my $contigs = '';
my $remote = '';
my $reads = '';

my $output = '';
my $reference = '';
my $runid = int(rand(10000)); 

my $readqc_options = "";
my $unitig_options = "";
my $mapping_options = "";
my $assembly_options = "";
my $annotation_options = "";
my $assemblyqc_options = "";
my $summary_options = "";

my $unitigs_off = 0;
my $read_qc_off = 0;
my $mapping_off = 0;
my $annotation_off = 0;
my $assembly_off = 0;
my $assembly_qc_off = 0;

my $keep = 0;
my $keep_remote = 0;
my $cpus = 1;
my $force = 0;
my $test = 0;
my $help = 0;

GetOptions(
		
	'contigs=s' => \$contigs,
	'reads=s' => \$reads,
	'remote=s' => \$remote,
	'output=s' 	=> \$output,
	
	'reference=s' => \$reference,
	'runid=i' => \$runid,
	 
	'readqc-options=s' => \$readqc_options,
	'unitig-options=s' => \$unitig_options,
      'summary-options=s' => \$summary_options,
	
	'no-readqc' => \$read_qc_off,
	'no-unitigs' => \$unitigs_off,
	
	'keep-intermediate=i' => \$keep, 
	'keep-remote' => \$keep_remote,
	'cpus=i' => \$cpus,
	'force' => \$force,
	'test' => \$test,
	'help|?' => \$help,
			
) or pod2usage(1);
pod2usage(1) if $help;

# set test paths
if ( $test == 1 ){

	# make test list for contigs
	#$contigs = "$script_path/../test/contig_list.txt";
	#`echo "test_ctg\t$script_path/../test/example.fna" > $contigs`;
	
	# make test list for read
	$reads = "$script_path/../test/list.txt";
    	`echo "test_reads\t$script_path/../test/reads_R1.fastq.gz\t$script_path/../test/reads_R2.fastq.gz\n" > $reads`;
	
	# set test list for remote 
	#$remote = "$script_path/../test/remote_test.tab";
	
	# general settings 
	$output = "$script_path/../test/test";
	$reference = "$script_path/../test/example.fna";
	$runid = 1;
	$force = 1;

}

# check inputs
my $input_check = 0;
for ($contigs, $remote, $reads){ ++$input_check if $_ ne '' }
pod2usage( {-message => q{ - ERROR requires at least one input arguement --remote, --reads or --contigs}, -exitval => 1, -verbose => 1 } ) if $input_check == 0; 
pod2usage( {-message => q{ - ERROR: output path is a required arguement}, -exitval => 1, -verbose => 1 } ) if $output eq '';

# check lists for redundant file names and missing files
my $errors = 0;
my %all_samples = ();
for my $file ($contigs, $remote, $reads){ 
	
	if ($file ne ''){
	
		# check file is present
		die " - ERROR: could not find $file\n" if !(-f $file);
		
		# store sample names
		my %samples = ();		
		my $red = 0;
		 
		# check path to files are present and store sample names
		my $count = 0;
		open F, $file or die " - ERROR: could not open $file\n";
		while(<F>){
					
			my $line = $_;
			chomp $line;
			
			if (/\S+/){
			
				++$count;
			
				my @f = split(/\t/, $line, -1);
				
				# check for number of entries
								
				# store sample names
				$samples{$f[0]}++;
				$all_samples{$f[0]}++;
			
				# check files present
				if ($file eq $contigs){
			
					if (@f != 2){
						print " - WARNING: incorrect number of entries in $file, expecting sample/contigs - $line\n";
						++$errors;
					}
				
					unless( -f $f[1] ){
						print " - WARNING: could not find $f[1] from input $file\n";
						++$errors;
					}
				}elsif ($file eq $reads){
				
					if (@f != 3){
						print " - WARNING: incorrect number of entries in $file, expecting sample/R1/R2 - $line\n";
						++$errors;
					}
				
					unless( -f $f[1] ){
						print " - WARNING: could not find $f[1] from input $file\n";
						++$errors;
					}
					unless( -f $f[2] ){
						print " - WARNING: could not find $f[2] from input $file\n";
						++$errors;
					}
				}elsif ($file eq $remote){
			
					if (@f != 2){
						print " - WARNING: incorrect number of entries in $file, expecting sample name/accession - $line\n";
						++$errors;
					}
						
				}
				
			}
		
		}close F;
		
		# no samples in file
		if ($count == 0){
			print " - WARNING: no samplepaths in $file\n";
			++$errors;
		}
		
		# check redundancies
		for my $k (keys %samples){
			if($samples{$k} > 1){
				print " - WARNING: redundant sample name $k in $file\n";
				++$errors;
			}
		}
	}

}

# check redundancies
for my $k (keys %all_samples){
	if($all_samples{$k} > 1){
		print " - WARNING: redundant sample name $k in collection\n";
		++$errors;
	}
}

# die if errors in inputs
die " - ERROR: errors found in input file(s), see above. Correct before continuing.\n" if $errors > 0;

# check output directory
unless ( -e $output ){
	die " - ERROR: could not make output directory ($output)\n" unless mkdir($output); 
}
$output = abs_path($output);

# feedback
print "\n\n\----------------------------------------------------------\n\n";
print " - run id: $runid\n";
print "\n----------------------------------------------------------\n\n";

# options
my $parse_opts = "";
$parse_opts = $parse_opts." -f" if $force == 1;
$parse_opts = $parse_opts." --general-options \'--keep $keep\'" if (($keep == 1) || ($keep == 2));
$parse_opts = $parse_opts." --ref $reference" if $reference ne "";

$parse_opts = $parse_opts." --readqc-options \"$readqc_options\"" if $readqc_options ne "";
$parse_opts = $parse_opts." --unitig-options \"$unitig_options\"" if $unitig_options ne "";
$parse_opts = $parse_opts." --mapping-options \"$mapping_options\"" if $mapping_options ne "";
$parse_opts = $parse_opts." --assembly-options \"$assembly_options\"" if $assembly_options ne "";
$parse_opts = $parse_opts." --annotation-options \"$annotation_options\"" if $annotation_options ne "";
$parse_opts = $parse_opts." --assemblyqc-options \"$assemblyqc_options\"" if $assemblyqc_options ne "";

# module toggle
$parse_opts = $parse_opts." --no-readqc" if $read_qc_off == 1;
$parse_opts = $parse_opts." --no-unitigs" if $unitigs_off == 1;
$parse_opts = $parse_opts." --no-mapping" if $mapping_off == 1;
$parse_opts = $parse_opts." --no-assembly" if $assembly_off == 1;
$parse_opts = $parse_opts." --no-annotation" if $annotation_off == 1;
$parse_opts = $parse_opts." --no-assemblyqc" if $assembly_qc_off == 1;

# number of samples for batching 
my $batch_no = $cpus * 3;

# parse contigs, reads and remote files seperately

# working input types
my %wtypes = ();
$wtypes{"contigs"} = 1 if $contigs ne "";  
$wtypes{"reads"} = 1 if $reads ne "";  
$wtypes{"remote"} = 1 if $remote ne "";  
my @wtypes = sort(keys(%wtypes));

# parse remote
for my $type (@wtypes){

	my $wfile = "";
	if ( $type eq "remote"){
		$wfile = $remote;
	}elsif ( $type eq "contigs" ){ 
		$wfile = $contigs
	}elsif ( $type eq "reads" ){ 
		$wfile = $reads;
	}
	
	# feedback 
	print " - processing $type:\n";
	
	# check number of files in the list 
	my $files = `wc -l < $wfile`;
	chomp $files;
	
	# remove previous temp files 
	unlink glob "$output/wgs_temp*";
	
	# expected number of batches
	my $b = ceil($files/$batch_no);
		
	# parse list to series of commands
	system("$script_path/parse_list.pl --input $wfile --$type --output $output --runid \"$runid\_$type\" $parse_opts");
		
	# split command list into batches
	if ($files>$cpus){
		`split -a 4 -d -l $batch_no $output/$runid\_$type.commands.txt $output/wgs_temp`;
	}else{
		`cp $output/$runid\_$type.commands.txt $output/wgs_temp0000`;
	}
	
	# set up feedback 
	print " - 0% processed    ";
	
	for my $i (0..$b-1){
	
		# set working command file 
		my $wc = sprintf("$output/wgs_temp%.4d", $i);
		
		# sanity check 
		die " - ERROR: could not find expected run file - $wc\n" unless ( -f $wc );	
			
		# run wgs pipe
		system("parallel -j $cpus < $wc");
	
		# feedback
		my $per_processed = sprintf( "%.2f", (($i+1)/$b) * 100 );
		print "\r - $per_processed% processed     ";
				
	
	}
	
	# clean up 
	unlink glob "$output/wgs_temp*";
	
	# feedback
	print "\r - 100\% processed    \n";
	print "\n----------------------------------------------------------\n\n";
	
}

# concatenate sample files
my $summary_input = "$output/summary_list.$runid";
system("echo -n \"\" > $summary_input");
system("cat $summary_input $contigs > $output/temp.$runid.tab && mv $output/temp.$runid.tab $summary_input") if $contigs ne "";
system("cat $summary_input $reads > $output/temp.$runid.tab && mv $output/temp.$runid.tab $summary_input") if $reads ne "";
system("cat $summary_input $remote > $output/temp.$runid.tab && mv $output/temp.$runid.tab $summary_input") if $remote ne "";

# summarise run qc
my $summary_dir = "$output/results_$runid/";
unless ( -e $summary_dir ){
	die " - ERROR: could not make output directory ($summary_dir)\n" unless mkdir($summary_dir); 
}
unless  ( $summary_options ne ""){
	system("perl $script_path/summarise_set.pl --input $summary_input --results $output --output $summary_dir --prefix $runid");
	die " - ERROR: summarise_set failed\n" if $?;
}else{
	my @summary_opts = split(/\s+/, $summary_options);
	my $summary_in = join( " ", @summary_opts);

	# make summary command
	my $summary_command = "perl $script_path/summarise_set.pl --input $summary_input --results $output --output $summary_dir --prefix $runid $summary_in";

	system($summary_command);
	die " - ERROR: summarise_set failed\n" if $?;
}

exit;

=cut

# parse contigs 
if ($contigs ne ""){

	# feedback 
	print "Processing contig files:\n";

	# parse contig file for commands
	system("perl $script_path/parse_list.pl --input $contigs --output $output --runid \"$runid\_contigs\" --remote $parse_opts");
	die " - ERROR: parse_list failed for $contigs\n" if $?;
	
	# check number of files in the list 
	my $files = `wc -l < $output/$runid\_contigs.commands.txt`;
		
	# check for warnings 
	my $no_warnings = 0;
	$no_warnings = `wc -l < $output/$runid\_contigs.warnings.txt` if -f "$output/$runid\_contigs.warnings.txt";
	chomp $no_warnings;
	print " - WARNING: $no_warnings warnings when processing input lists - see $output/$runid\_contigs.warnings.txt\n" if $no_warnings > 0;

	# set up feedback 
	print " - 0% processed    ";
					
	# make temp file for parallelised commands
	open TEMP, ">$output/$runid.temp" or die " - ERROR: could not open temp file ($output./$runid.temp)\n";
	
	# parse command file and run wgs-pipe
	if ($files > 0){
	
		my $count = 0;
		my $temp_count = 0; 
		open COMMANDS, "$output/$runid\_contigs.commands.txt" or die " - ERROR: could not open command file ($output/$runid\_contigs.commands.txt)\n";
		while (<COMMANDS>) {
	
			++$count;
			++$temp_count;
			
			# identify sample name
			my $line = $_;
			chomp $line;
			my @line =  split(/\t/, $line);			
			my $s_name = $line[0];		
			
			# print command line
			print TEMP "$line[1]\n";
			
			# make sample directory
			unless ( -e "$output/$s_name" ){
				die " - ERROR: could not make output directory ($output/$s_name)\n" unless mkdir("$output/$s_name"); 
			}

			if ( ( $count == $files ) || ( $temp_count == $batch_no) ){
				
				# close temp file
				close TEMP; 
				
				# run wgs-pipe
				system("parallel -j $cpus < $output/$runid.temp");
				
				# reset variables
				open TEMP, ">$output/$runid.temp" or die " - ERROR: could not open temp file ($output./$runid.temp)\n";
				$temp_count = 0;
				
				# feedback
				my $per_processed = sprintf( "%.2f", ($count/$files) * 100 );
				print "\r - $per_processed\%      ";
				
			}		
	
		}
		
	}
	
	# clean up 
	unlink("$output/$runid.temp") if -f "$output/$runid.temp";
	
	# feedback
	print "\r - 100\% processed    \n";
	print "\n----------------------------------------------------------\n\n";
}

# parse reads
if ($reads ne ""){

	# feedback 
	print "Processing read files:\n";

	# parse contig file for commands
	system("perl $script_path/parse_list.pl --input $reads --output $output --runid \"$runid\_reads\"  $parse_opts");
	die " - ERROR: parse_list failed for $reads\n" if $?;
	
	# check number of files in the list 
	my $files = `wc -l < $output/$runid\_reads.commands.txt`;
		
	# check for warnings 
	my $no_warnings = 0;
	$no_warnings = `wc -l < $output/$runid\_reads.warnings.txt` if -f "$output/$runid\_reads.warnings.txt"; 
	chomp $no_warnings;
	print " - WARNING: $no_warnings warnings when processing input lists - see $output/$runid\_reads.warnings.txt\n" if $no_warnings > 0;

	# set up feedback 
	print " - 0% processed    ";
					
	# make temp file for parallelised commands
	open TEMP, ">$output/$runid.temp" or die " - ERROR: could not open temp file ($output./$runid.temp)\n";
	
	# parse command file and run wgs-pipe
	if ($files > 0){
	
		my $count = 0;
		my $temp_count = 0; 
		open COMMANDS, "$output/$runid\_reads.commands.txt" or die " - ERROR: could not open command file ($output/$runid\_reads.commands.txt)\n";
		while (<COMMANDS>) {
	
			++$count;
			++$temp_count;
		
			# identify sample name
			my $line = $_;
			chomp $line;
			my @line =  split(/\t/, $line);			
			my $s_name = $line[0];		
			
			# print command line
			print TEMP "$line[1]\n";
			
			# make sample directory
			unless ( -e "$output/$s_name" ){
				die " - ERROR: could not make output directory ($output/$s_name)\n" unless mkdir("$output/$s_name"); 
			}

			if ( ( $count == $files ) || ( $temp_count == $batch_no) ){
				
				# close temp file
				close TEMP; 
				
				# run wgs-pipe
				system("parallel -j $cpus < $output/$runid.temp");
				
				# reset variables
				open TEMP, ">$output/$runid.temp" or die " - ERROR: could not open temp file ($output./$runid.temp)\n";
				$temp_count = 0;
				
				# feedback
				my $per_processed = sprintf( "%.2f", ($count/$files) * 100 );
				print "\r - $per_processed\%      ";
				
			}		
	
		}
		
	}
	
	# clean up 
	unlink("$output/$runid.temp") if -f "$output/$runid.temp";
	
	# feedback
	print "\r - 100\% processed    \n";
	print "\n----------------------------------------------------------\n\n";
	
}



# concatenate sample files
my $summary_input = "$output/summary_list.$runid";
system("echo -n \"\" > $summary_input");
system("cat $summary_input $contigs > $output/temp.$runid.tab && mv $output/temp.$runid.tab $summary_input") if $contigs ne "";
system("cat $summary_input $reads > $output/temp.$runid.tab && mv $output/temp.$runid.tab $summary_input") if $reads ne "";
system("cat $summary_input $remote > $output/temp.$runid.tab && mv $output/temp.$runid.tab $summary_input") if $remote ne "";

# summarise run qc
my $summary_dir = "$output/results/";
unless ( -e $summary_dir ){
	die " - ERROR: could not make output directory ($summary_dir)\n" unless mkdir($summary_dir); 
}
unless  ( $summary_options ne ""){
	system("perl $script_path/summarise_set.pl --input $summary_input --results $output --output $summary_dir --prefix $runid");
	die " - ERROR: summarise_set failed\n" if $?;
}else{
	my @summary_opts = split(/\s+/, $summary_options);
	my $summary_in = join( " ", @summary_opts);

	# make summary command
	my $summary_command = "perl $script_path/summarise_set.pl --input $summary_input --results $output --output $summary_dir --prefix $runid $summary_in";

	system($summary_command);
	die " - ERROR: summarise_set failed\n" if $?;
}

# clean up if test
if ($test == 1){
	
	# clean up settings files
	unlink $contigs;
	unlink $reads; 
}

exit
