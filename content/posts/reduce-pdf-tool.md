+++
title = 'Reduce PDF Tool'
author = 'Mochan Shrestha'
date = 2024-01-07T08:09:34-05:00
draft = false
+++

This is a tool to reduce the size of PDF files by removing or compressing images. It can also be asked to only output a section of pages from the original PDF file.

When uploading PDFs for LLMs, the file size is usually limited. Some PDFs contain a large number of high quality images that bump up the file size.

Whenever this happened, I would just copy/paste the text. But, I don't know if the PDF processing on the server is getting better and that the visual context of the PDF is also being used. So, an easy solution is to have a tool that can reduce the size of the PDF file by compressing the images since high quality images are probably not necessary.

![Reduce-PDF Icon](/images/reduce-pdf-icon.png)

Source code is available on [GitHub](https://github.com/mochan-b/reduce-pdf) and the PyPi package is available [here](https://pypi.org/project/reduce-pdf/).

## Usage

```bash
$ reduce-pdf <input-file> <output-file> [options]
```

### Options

- '--remove': Remove images from the PDF file.
- '--compress': Compress images in the PDF file with the given JPEG quality. The quality is a number between 0 and 100. The default is 0.
- '--start_page': The first page to output. The default is 1.
- '--end_page': The last page to output. The default is the last page of the input file.

## Implementation

The tool is implemented in Python using the [PyPDF2](https://pypi.org/project/PyPDF2/).

A PDF file contains a collection of objects. When removing images, we just need to find the object and delete it.

```python
# Open the PDF file
doc = fitz.open(pdf_file_path)

# Iterate through the pages
for page_no in range(len(doc)):
    page = doc.load_page(page_no)
    # Iterate through the images and delete them
    img_list = page.get_images(full=True)
    for img in img_list:
        xref = img[0]
        doc._deleteObject(xref)

# Save the changes
doc.save("images_removed.pdf", garbage=4, deflate=True, clean=True)
doc.close()
```

When compressing images, we need to find the images, get the image data, compress it, put it in the same place as the original image and then delete the original image.

```python
# Open the original PDF file
doc = fitz.open(pdf_file_path)

# Iterate through the pages
for page_no in range(len(doc)):
    page = doc.load_page(page_no)

    # Get list of images on the page
    img_list = page.get_images(full=True)

    for img in img_list:
        xref = img[0]
        base_image = doc.extract_image(xref)
        img_bytes = base_image["image"]

        # Open the image using PIL
        image = Image.open(io.BytesIO(img_bytes))

        # Compress the image - adjust quality as needed
        with io.BytesIO() as output:
            image.save(output, format='JPEG', quality=25)  # Adjust quality for higher compression
            compressed_image_bytes = output.getvalue()

        # Replace the original image with the compressed one
        image_rect = page.get_image_rects(xref)[0]

        # Delete the original image from the PDF
        doc._deleteObject(xref)

        page.insert_image(rect=image_rect, stream=compressed_image_bytes)

# Save the changes to a new file
doc.save("images_compressed.pdf", garbage=4, deflate=True, clean=True)
doc.close()
```

When removing pages, if we just delete the pages the pages go away but the internal contents of the page is still in the PDF file and the file size is not reduced. So, we have to go into usused pages and delete all of the images there.

The full script for the tool is available [here](https://github.com/mochan-b/reduce-pdf/blob/main/reduce_pdf/reduce_pdf.py)

## Limitations

- The tool compresses everything to JPEG and so the transparency is lost if the original image was PNG. It currently replaces it with a black background.