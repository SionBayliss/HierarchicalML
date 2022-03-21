################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../bloom_filter.cpp \
../fastq.cpp \
../hash.cpp \
../kmer.cpp \
../kmerintpair.cpp \
../kseq.cpp \
../main.cpp \
../message.cpp \
../option.cpp \
../paracorrect.cpp \
../thread.cpp \
../voting.cpp 

OBJS += \
./bloom_filter.o \
./fastq.o \
./hash.o \
./kmer.o \
./kmerintpair.o \
./kseq.o \
./main.o \
./message.o \
./option.o \
./paracorrect.o \
./thread.o \
./voting.o 

CPP_DEPS += \
./bloom_filter.d \
./fastq.d \
./hash.d \
./kmer.d \
./kmerintpair.d \
./kseq.d \
./main.d \
./message.d \
./option.d \
./paracorrect.d \
./thread.d \
./voting.d 


# Each subdirectory must supply rules for building sources it contributes
%.o: ../%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"   -fopenmp -I $(OUTPUT_PREFIX}
	@echo 'Finished building: $<'
	@echo ' '


