def:
	@echo USAGE: "make cuda" or "make sycl"

cuda:
	$(MAKE) -f Makefile.cuda

sycl:
	$(MAKE) -f Makefile.sycl

clean:
	@rm -rf bin build include lib
