set.seed(1234)
options(digits = 3)

knitr::opts_chunk$set(
    comment = "#>",       # characters to show in the output box
    collapse = FALSE,     # output will not be shown; figures will show
    # collapse = TRUE,    # output will not be shown; figures will show
    results = "hold",     # wait for last operation before printing
    cache = FALSE,
    out.width = "70%",
    fig.align = 'center',
    fig.width = 6,
    fig.asp = 0.618,  # 1 / phi
    fig.show = "hold"
)

# options(dplyr.print_min = 6, dplyr.print_max = 6)
