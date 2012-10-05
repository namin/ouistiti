# Turn a .csv file into a .lua file in the style of 
# http://www.lua.org/pil/12.html
# e.g. the line `0,1,2` is turned into the line `Entry{0,1,2}`

BEGIN {
    # remove header line
    getline
}
{
    printf("Entry{%s}\n", $0)
}