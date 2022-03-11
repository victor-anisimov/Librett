def:
	@echo USAGE: "make cuda", "make hip" or "make sycl"

cuda:
	$(MAKE) -f Makefile.cuda

amd:
	$(MAKE) -f Makefile.amd

hip:
	$(MAKE) -f Makefile.hip

sycl:
	$(MAKE) -f Makefile.sycl

clean:
	@rm -rf bin build include lib
