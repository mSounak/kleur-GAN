# kleur-GAN
Colourizing black and white images, using pix2pix gan architecture.

## Dataset
Used flicker8k dataset.
  * 6k training set
  * 2k testing set
Transformed the images in the dataset to L*ab colour format and used L (lightness) channel as input and ab (colour) channel as output.

## Training
Trained the model for 100 epochs.

## Result

![results](https://github.com/mSounak/kleur-GAN/blob/main/backend/generated_images/results.png)
 1st row is the input (black and white image), 2nd row is the predicted image and 3rd row is the original image.
