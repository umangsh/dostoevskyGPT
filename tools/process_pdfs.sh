#!/bin/zsh

# Process a single pdf
process_pdf() {
    filename="$1"
    f="$2"  # first page to convert
    l="$3"  # last page to convert
    x="${4:-0}"  # x-coordinate of the crop area top left corner
    y="${5:-0}"  # y-coordinate of the crop area top left corner
    W="${6:-0}"  # width of crop area in pixels
    H="${7:-0}"  # height of crop area in pixels

    local input_filepath="books/${filename}.pdf"
    local output_filepath="books_txt/${filename}.txt"
    local temp_filepath="tmp.txt"

    pdftotext -f "$f" -l "$l" -x "$x" -y "$y" -W "$W" -H "$H" "$input_filepath" "$output_filepath"
    tr -d '\f' <"$output_filepath" >"$temp_filepath" && mv "$temp_filepath" "$output_filepath"
}

# List of all files to be processed

# No post-processing required

#process_pdf "An-Honest-Thief" "1" "28" "50" "50" "600" "700"
#process_pdf "An-Unpleasant-Predicament" "3" "49"
#process_pdf "Bobok-From-Somebodys-Diary" "3" "17"
#process_pdf "Crime-and-Punishment" "4" "404" "0" "80" "600" "620"
#process_pdf "The-Brothers-Karamazov" "7" "695" "0" "75" "520" "600"
#process_pdf "Notes-from-the-Underground" "6" "89" "0" "80" "600" "620"
#process_pdf "Polzunkov" "4" "17"
#process_pdf "The-Adolescent-by-Fyodor-Dostoevsky" "31" "686" "0" "0" "600" "725"
#process_pdf "The-Eternal-Husband" "5" "112"
#process_pdf "The-Peasant-Marey" "3" "7"
#process_pdf "The-Possessed" "6" "706" "0" "50" "600" "730"
#process_pdf "White-Nights-And-Other-Stories" "5" "295" "0" "50" "600" "730"

# Post-processing required

#process_pdf "A-Christmas-Tree-and-a-Wedding" "3" "9"  # needs post-processing
#process_pdf "A-Faint-Heart" "4" "42"  # needs post-processing
#process_pdf "A-Little-Hero" "4" "33"  # needs post-processing
#process_pdf "A-Novel-in-Nine-Letters" "3" "13"  # needs post-processing
#process_pdf "Another-Mans-Wife" "3" "42"  # needs post-processing
#process_pdf "Gentle-Spirit" "5" "51"  # needs post-processing
#process_pdf "Mr-Prohartchin" "3" "29"  # needs post-processing
#process_pdf "Poor-Folk" "1" "64"   # needs post-processing
#process_pdf "The-Crocodile" "3" "33"  # needs post-processing
#process_pdf "The-Double" "5" "161"  # needs post-processing
#process_pdf "The-Dream-of-a-Riciculus-Man" "1" "13"  # needs post-processing
#process_pdf "The-Gambler" "1" "158" "0" "0" "600" "725"  # needs post-processing
#process_pdf "The-Heavenly-Christmas-Tree" "3" "6"  # needs post-processing
#process_pdf "The-House-of-the-Dead" "9" "308"  # needs post-processing
#process_pdf "The-Idiot" "1" "1149" "0" "50" "600" "425"  # needs post-processing
#process_pdf "The-Insulted-and-the-Injured" "7" "404"  # needs post-processing

# Lots of post-processing required.

#process_pdf "Buried-Alive" "4" "364"  # needs post-processing ***
#process_pdf "Letters-of-Fyodor-Michailovitch-Dostoevsky-to-his-Family-and-Friends" "15" "343"  # needs post-processing ***
