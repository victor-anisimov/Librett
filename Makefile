def:
	@echo ""
	@echo "USAGE: make platform [option=-DPERFTEST]"
	@echo "            platform = cuda, hip, or sycl"
	@echo "            option=-DPERFTEST compiles the light version"
	@echo "                of librett_test for performance analysis"
	@echo "EXAMPLE: make hip option=-DPERFTEST"
	@echo ""

cuda:
	$(MAKE) -f Makefile.cuda option=$(option)

hip:
	$(MAKE) -f Makefile.hip  option=$(option)

complex:
	$(MAKE) -f Makefile.complex  option=$(option)

sycl:
	$(MAKE) -f Makefile.sycl option=$(option)

clean:
	@rm -rf bin build include lib *.d

