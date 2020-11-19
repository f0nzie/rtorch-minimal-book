OUTPUT_DIR = .
BOOKDOWN_FILES = _bookdown_files
PKGNAME = `sed -n "s/Package: *\([^ ]*\)/\1/p" DESCRIPTION`
PKGVERS = `sed -n "s/Version: *\([^ ]*\)/\1/p" DESCRIPTION`
# Detect operating system. Sort of tricky for Windows because of MSYS, cygwin, MGWIN
ifeq ($(OS), Windows_NT)
    OSFLAG = WINDOWS
else
    UNAME_S := $(shell uname -s)
    ifeq ($(UNAME_S), Linux)
        OSFLAG = LINUX
    endif
    ifeq ($(UNAME_S), Darwin)
        OSFLAG = OSX
    endif
endif


.PHONY: render-book
render-book:
	Rscript -e "bookdown::render_book(input='.', output_format = 'bookdown::gitbook', config_file='_bookdown.yml')"



# delete unwanted files and folders in bookdown folder
.PHONY: tidy
tidy:
	find $(OUTPUT_DIR) -maxdepth 1 -name \*.md -delete
	find $(OUTPUT_DIR) -maxdepth 1 -name \*.rds -delete
	find $(OUTPUT_DIR) -maxdepth 1 -name \*.ckpt -delete
	find $(OUTPUT_DIR) -maxdepth 1 -name \*.nb.html -delete
	if [ -d ${BOOKDOWN_FILES} ]; then rm -rf ${BOOKDOWN_FILES};fi \


