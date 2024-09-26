#
#     Overall Makefile for ADASS GUI Tutorial Examples
#
#  This makefile should be at the Vulkan or Metal level for
#  the set of example programs. It just runs 'make' on all
#  the three example programs, which should be in the Adder,
#  Median and Mandel sub-directories.
#
#  The default target builds all the example programs, and
#  the 'clean' target cleans them all back to the source files.

Target : MakeAdder MakeMedian MakeMandel

MakeAdder :
	@echo Making Adder
	@cd Adder && $(MAKE)

MakeMedian :
	@echo Making Median
	@cd Median && $(MAKE)

MakeMandel :
	@echo Making Mandel
	@cd Mandel && $(MAKE)

clean : CleanAdder CleanMedian CleanMandel

CleanAdder :
	@echo Cleaning Adder
	@cd Adder && $(MAKE) clean

CleanMedian :
	@echo Cleaning Median
	@cd Median && $(MAKE) clean

CleanMandel :
	@echo Cleaning Mandel
	@cd Mandel && $(MAKE) clean

