
# Just a place for some Source Engine tools

## GMA/VPK
Drag & drop your `.gma` or `.vpk` file onto the matching script to extract it.

- `gma_extractor.py`: extract `.gma` addons
- `vpk_extractor.py`: extract Valve `.vpk` archives

Extracted files are written to a folder next to the original archive (same directory, folder named after the archive).

If you have a `<filename>-legacy.bin` file instead of a `.gma`
- unzip it with 7zip
- change the extracted file's extension to `.gma`

## VTF
- `vtf_browser.py`: browse dir&subdirs for `.vmt` files and display thumbnails/metadata
- - `pip install PyQt5 vtf2img `


## MDL
Very WIP, only really useful for seeing some basic info for now
- `mdl_browser.py`: browse dir&subdirs for `.mdl` files and display some basic info about them


