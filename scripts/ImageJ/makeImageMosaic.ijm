/*
 * @title: Marco for reating mosaic from multiple images
 * @author: Jiri Borovec
 * @date: 13/02/2013
 * @mail: jiri.borovec@fel.cvut.cz
 * 
 * @brief: This macro does mosaic from all open images. 
 * The grid size is given by param "mosaicSize".
 * It keeps the final image ans the others will be closed.
 */

//http://imagej.nih.gov/ij/developer/macro/functions.html

// size of mosaic grid
mosaicSize = 50;

run("Images to Stack", "method=[Copy (center)] name=Stack title=[]");
//run("Stack to Images");
selectWindow("Stack");
// get information about the images
width = getWidth();
height = getHeight();
nbImgs = nSlices;
// create new image for the mosaic
newImage("Mosaic", "RGB White", width, height, 1);

yIdx = 0;
sMosaicY = mosaicSize;
// going throw the image in vertical direction
for(y=0; y<height; y+=mosaicSize) {
	//print(y);
  	xIdx = yIdx;
  	// if we are close to image border make the rectangle smaller
  	if ((y+mosaicSize)>height) {
  		sMosaicY = height - y;
  	}
  	sMosaicX = mosaicSize;
  	// going throw the image in horizontal direction
  	for(x=0; x<width; x+=mosaicSize) {
  		// if we are close to image border make the rectangle smaller
  		if ((x+mosaicSize)>width) {
  			sMosaicX = width - x;
  		}
  		selectWindow("Stack");
		Stack.setSlice(xIdx+1);
		makeRectangle(x, y, sMosaicX, sMosaicY);
		run("Copy");
		selectWindow("Mosaic");
		makeRectangle(x, y, sMosaicX, sMosaicY);
		run("Paste");
		xIdx = (xIdx+1) % nbImgs;
  	} 
  	yIdx = (yIdx+1) % nbImgs;
} 

//close("Stack");
