OUTPUT_DIR = .
BOOKDOWN_FILES = _bookdown_files
PKGNAME = `sed -n "s/Package: *\([^ ]*\)/\1/p" DESCRIPTION`
PKGVERS = `sed -n "s/Version: *\([^ ]*\)/\1/p" DESCRIPTION`


.PHONY: rendersite
rendersite:
	Rscript -e "bookdown::render_book(input='.', output_format = 'bookdown::gitbook', config_file='_bookdown.yml')"



# delete unwanted files and folders in bookdown folder
.PHONY: tidy
tidy:
	find $(OUTPUT_DIR) -maxdepth 1 -name \*.md -delete
	find $(OUTPUT_DIR) -maxdepth 1 -name \*.rds -delete
	find $(OUTPUT_DIR) -maxdepth 1 -name \*.ckpt -delete
	find $(OUTPUT_DIR) -maxdepth 1 -name \*.nb.html -delete
	if [ -d ${BOOKDOWN_FILES} ]; then rm -rf ${BOOKDOWN_FILES};fi \


