import ImageConnector


address = "C:\\Users\\uqmbonya\\Downloads\\tiny-imagenet-200\\tiny-imagenet-200\\val\\images\\"


img = ImageConnector.read_images(address, "JPEG", [10, 10], True)
