import ImageConnector
import CnnVisualiser

# data_address = "R:\\projects\\image-net\\tiny-imagenet-200\\tiny-imagenet-200\\val\\images\\"
data_address = "C:\\Users\\vardi\\Documents\\Datasets\\tiny-imagenet-200\\tiny-imagenet-200\\val\\images\\"

(imgs, trans) = ImageConnector.read_images(data_address, "JPEG", [64, 64], "k-space", True)

# datafile = cbook.get_sample_data('ada.png')
# h = Image.open(datafile)
#
# dpi = rcParams['figure.dpi']
layers = {"type": [], "images": [], "info": []}
layers["type"].append("Conv")
layers["type"].append("Conv")
layers["type"].append("Flat")
layers["type"].append("Conv")
layers["type"].append("Conv")

layers["images"].append(imgs[0:4])
layers["images"].append(imgs[19:49])
layers["images"].append(imgs[40:54])
layers["images"].append(imgs[30:35])
layers["images"].append(imgs[0:4])
layers["images"].append(imgs[20:24])

layers["info"].append("Conv: 10 filter, $5 \\times 5$")
layers["info"].append("Conv: 2 filters $5 \\times 50$")
layers["info"].append("Dense: $50 \\to 5$")
layers["info"].append("Conv: 2 filters $5 \\times 5$")
layers["info"].append("Conv: 2 filters $5 \\times 5$")


overlap_x = 0.15
overlap_y = 0.15
sv = 0.3
ev = 0.5
max_images = 9
buff_factor = .2
CnnVisualiser.vis_layers(layers, max_images, overlap_x, overlap_y, sv, ev, buff_factor)

# vis_layer(images, num_layers, max_images, layer_indx, overlap_x, overlap_y, sv, ev)
#
# layer_indx = 3
# vis_layer(images, num_layers, max_images, layer_indx, overlap_x, overlap_y, sv, ev)


