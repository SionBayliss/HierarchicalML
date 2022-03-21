#include <stdio.h>
#include <string.h>
#include <stdlib.h>

// This script randomly subsamples reads from paired fastq files to reach a desired coverage.
// To compile, type:
// gcc Subsample_reads.c -o Subsample_reads
// To run, type:
// Subsample_reads file_1 file_2 Reference_genome_length Desired_coverage
// optionally add 'f' as a fifth argument - fast mode (doesn't downsample and no new outputs are produced).
// You should include the full file path for the input files, and the output files will have 
// the same name, but will be appended with '.reduced'.
// Written by Harry Thorpe

int main( int argc, char *argv[] )
{

// Collect command line args.
long long int arg_len=1000;

//arg_len = strlen(argv[1]);
char in_file_1[arg_len];
strcpy(in_file_1, argv[1]);

//arg_len = strlen(argv[2]);
char in_file_2[arg_len];
strcpy(in_file_2, argv[2]);

//arg_len = strlen(argv[3]);
char ref_len_c[arg_len];
strcpy(ref_len_c, argv[3]);
long long int ref_len = atol(ref_len_c);

//arg_len = strlen(argv[4]);
char des_cov_c[arg_len];
strcpy(des_cov_c, argv[4]);
long long int des_cov = atol(des_cov_c);

int ss = 1;
if(argc > 5){
	int ret = strcmp(argv[5], "f");
	if(ret == 0){
		ss = 0;
		
		printf ("fast mode chosen - qc will be performed, but no subsampling will be done.\n");
	}
}

// Open file and count lines.
long long int line_count = 0;
long long int read_count = 0;
long long int read_len = 0;
long long int read_len_tot = 0;
long long int max_read_len = 0;

FILE *input = fopen (in_file_1, "r");
if(input != NULL){
	char line[1000];
	long long int line_len;
	while(fgets(line, sizeof line, input) != NULL){
		
		if(line_count % 4 == 1){
			read_count++;
			
			line_len = strlen(line);
			line_len = (line_len - 1);
			if(line[line_len] == '\n'){
				line[line_len] = '\0';
			}
			read_len = strlen(line);
			read_len_tot += read_len;
			
			if(read_len > max_read_len){
				max_read_len = read_len;
			}
		}
		line_count++;
	}
}
float read_len_ave = ((float)read_len_tot / (float)read_count);

// Estimate coverage (assuming perfect mapping).
float cov_ave = (((float)read_count * (float)read_len_ave * 2) / (float)ref_len);

printf ("Reference length = %lli\n", ref_len);
printf ("Number of reads = %lli\n", read_count);
printf ("Average read length = %f\n", read_len_ave);
printf ("Estimated coverage = %f\n", cov_ave);
printf ("Longest read = %lli\n", max_read_len);

if(cov_ave > des_cov && ss == 1){
	
	long long int *read_array = malloc(read_count * sizeof(long long int));
	long long int read = 0;
	for(read=0; read<read_count; read++){
		read_array[read] = read;
	}
	
	// Fisher-yates shuffle the array.
	long long int i, j, tmp, upper_bound;

	i = (read_count - 1);
	while(i > 0){
		upper_bound = RAND_MAX - ((RAND_MAX % (i + 1)) + 1);

		j = rand() % (i + 1);

		if(j <= upper_bound){
			tmp = read_array[j];
			read_array[j] = read_array[i];
			read_array[i] = tmp;
	
			i--;
		}
	}
	
	// Make array compatible with line array index.
	for(read=0; read<read_count; read++){
		read_array[read] = (read_array[read] * 4);
	}
	
	// Make array from line array indices.
	long long int *line_array = malloc(line_count * sizeof(long long int));
	long long int line;
	for(line=0; line<line_count; line++){
		line_array[line] = 0;
	}
	
	// Calculate number of reads needed.
	long long int reads_needed = (((ref_len * des_cov) / read_len_ave) / 2);
	printf ("%lli reads needed from %lli reads in original file.\n", reads_needed, read_count);
	
	for(read=0; read<reads_needed; read++){
		line_array[read_array[read]] = 1;
		line_array[(read_array[read]+1)] = 1;
		line_array[(read_array[read]+2)] = 1;
		line_array[(read_array[read]+3)] = 1;
	}
	
	char out_file_1[1000];
	strcpy(out_file_1, in_file_1);
	char ext[] = ".reduced";
	strcat(out_file_1, ext);
	FILE *output_1 = fopen (out_file_1, "w");
	
	line_count=0;
	FILE *input_1 = fopen (in_file_1, "r");
	if(input_1 != NULL){
		char line[1000];
		while(fgets(line, sizeof line, input_1) != NULL){
			if(line_array[line_count] == 1){
				fprintf (output_1, "%s", line);
			}
			line_count++;
		}
	}
	
	char out_file_2[1000];
	strcpy(out_file_2, in_file_2);
	strcat(out_file_2, ext);
	FILE *output_2 = fopen (out_file_2, "w");
	
	line_count=0;
	FILE *input_2 = fopen (in_file_2, "r");
	if(input_2 != NULL){
		char line[1000];
		while(fgets(line, sizeof line, input_2) != NULL){
			if(line_array[line_count] == 1){
				fprintf (output_2, "%s", line);
			}
			line_count++;
		}
	}
	
}

return 0;
}

