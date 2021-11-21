from blood_segmentation.generator import RetinalBloodVesselGenerator as Generator
from PIL import Image
from matplotlib import pyplot as plt


# Load generator
gen = Generator(net_type='unet', pretrained_model='./models/unet.pth')

# Load image
image_path = '00944015_fundus_0.png'
image = Image.open('sample/' + image_path)
image = image.convert(mode='RGB')

# Segment blood vessel and save to a file
gen.generate(image, 'output_' + image_path)

# Segment blood vessel and plot
output = gen.generate(image)
plt.imshow(output)
plt.show()