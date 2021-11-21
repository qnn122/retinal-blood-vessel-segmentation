echo 'Downloading and setting up models'
DEST_DIR='models'

echo 'Unet'
ggID='1iIRUzlDZkk2DQViMMWwx0JsqNn9YABXp'  
ggURL='https://drive.google.com/uc?export=download'  
FILENAME='unet.pth'
getcode="$(awk '/_warning_/ {print $NF}' /tmp/gcokie)"  
curl -Lb /tmp/gcokie "${ggURL}&confirm=${getcode}&id=${ggID}" -o "${FILENAME}"  
mkdir $DEST_DIR
mv $FILENAME $DEST_DIR
echo 'Done'

echo 'Dense UNet 4'
ggID='1kX9uGduxzP-ld3dj-Z-tsi_KP5M8Efgs'  
ggURL='https://drive.google.com/uc?export=download'  
FILENAME='dense_unet_4.pth'
getcode="$(awk '/_warning_/ {print $NF}' /tmp/gcokie)"  
curl -Lb /tmp/gcokie "${ggURL}&confirm=${getcode}&id=${ggID}" -o "${FILENAME}"  
mv $FILENAME $DEST_DIR
echo 'Done'

echo 'Dense UNet 5'
ggID='1lA-bg6Ks0GeC0rQpkbeZ8xw8kzDmw1Bq'  
ggURL='https://drive.google.com/uc?export=download'  
FILENAME='dense_unet_5.pth'
getcode="$(awk '/_warning_/ {print $NF}' /tmp/gcokie)"  
curl -Lb /tmp/gcokie "${ggURL}&confirm=${getcode}&id=${ggID}" -o "${FILENAME}"  
mv $FILENAME $DEST_DIR
echo 'Done'

echo 'Ladder Net'
ggID='1BONAA5mVqaChnAe9krhmBxF5NIYYx6aj'  
ggURL='https://drive.google.com/uc?export=download'  
FILENAME='laddernet.pth'
getcode="$(awk '/_warning_/ {print $NF}' /tmp/gcokie)"  
curl -Lb /tmp/gcokie "${ggURL}&confirm=${getcode}&id=${ggID}" -o "${FILENAME}"  
mv $FILENAME $DEST_DIR
echo 'Done'