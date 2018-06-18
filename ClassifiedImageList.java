import java.io.IOException;
import java.io.RandomAccessFile;
/**
 * Takes a file name as a string then reads the bytes into a list of 28x28 images.
 * The files it requires are the MNIST database records, specifically:
 * t10k-images.idx3-ubyte
 * t10k-labels.idx1-ubyte
 * train-images.idx3-ubyte
 * train-labels.idx1-ubyte
 * 
 * @author Bart Nadala
 * @version 1.0
 * @date March 19, 2017
 *
 */
public class ClassifiedImageList {
	public final int CLASSIFIERS = 17;
	public int[][][] images;
	private int length, x, y;
	private byte[] data, label;
	public double[][] vectors;
	
	public ClassifiedImageList(String filename){
		try {
			RandomAccessFile file = new RandomAccessFile(filename, "r");
			RandomAccessFile fileLabel = new RandomAccessFile(filename.replace("images.idx3-ubyte", "labels.idx1-ubyte"), "r");
		    try {
		    	data = new byte[(int)file.length()];
		    	label = new byte[(int)fileLabel.length()];
				file.readFully(data);
				fileLabel.readFully(label);
				length = readInt(4);
				x = readInt(8);
				y = readInt(12);
		    } finally {
		    	file.close();
		    	fileLabel.close();
		    }
		} catch (IOException ex) {
		    ex.printStackTrace();
		}
		
		//Load data and convert to binary
		images = new int[length][x][y];
		
		for (int n=0; n<length; n++){ 	
			for ( int i = 0, ij = 16 + n*x*y; i < y; i++){
				for ( int j = 0; j < x; j++){
					int d = data[ij++]&0xff;
					if(d<=128){
						images[n][i][j] = 0;
					}else{
						images[n][i][j] = 1;
					}
					
				}
			}
		}
		
		//Noise Removal, Collect area, height, width, width-height ratio, and distribution features.
		vectors = new double[length][CLASSIFIERS];
		int[][] temp = new int[x][y];
		for(int n=0; n<length; n++){	
			for(int i=1; i<x-1; i++){
				for(int j=1; j<y-1; j++){
					if(images[n][i][j]==1){
						int count=0;
						int[] line = new int[9];
						line[0] = images[n][i][j];
						line[1] = images[n][i-1][j];
						line[2] = images[n][i-1][j+1];
						line[3] = images[n][i][j+1];
						line[4] = images[n][i+1][j+1];
						line[5] = images[n][i+1][j];
						line[6] = images[n][i+1][j-1];
						line[7] = images[n][i][j-1];
						line[8] = images[n][i-1][j-1];
						for(int k=0; k<9; k++){
							count+=line[k];
						}
						if(count<=2){
							temp[i][j] = 0;
						}else if(count==3 &&(line[0]*line[3]*line[4]==1 ||
								line[0]*line[1]*line[3]==1 ||
								line[0]*line[1]*line[8]==1 ||
								line[0]*line[7]*line[8]==1 ||
								line[0]*line[1]*line[7]==1 ||
								line[0]*line[6]*line[7]==1 ||
								line[0]*line[5]*line[7]==1 ||
								line[0]*line[4]*line[5]==1)){
							temp[i][j]=0;
						}else if(count==4 && (line[0]*line[1]*line[7]*line[8]==1 ||
								line[0]*line[1]*line[2]*line[3]==1)){
							temp[i][j]=0;
						}else{
							temp[i][j]=1;
						}
					}else{
						temp[i][j]=0;
					}
				}
			}
			vectors[n][0]=0;
			vectors[n][4] = 0;
			vectors[n][5] = 0;
			vectors[n][6] = 0;
			vectors[n][7] = 0;			
			int wLeft=y, wRight=0;
			int hTop=x, hBottom=0;
			for(int i=1; i<x-1; i++){
				for(int j=1; j<y-1; j++){
					images[n][i][j] = temp[i][j];
					if(images[n][i][j]==1){
						vectors[n][0]++;
						if(j>wRight)
							wRight=j;
						if(j<wLeft)
							wLeft = j;
						if(i>hBottom)
							hBottom = i;
						if(i<hTop)
							hTop=i;
						if(j<(y/2)){
							if(i<(x/2))
								vectors[n][4]++;
							else
								vectors[n][6]++;
						}else
							if(i<(x/2))
								vectors[n][5]++;
							else
								vectors[n][7]++;
					}
				}
			}
			vectors[n][1] = 1 + hBottom - hTop;
			vectors[n][2] = 1 + wRight - wLeft;
			if(vectors[n][1]!=0)
				vectors[n][3] = (1.0*vectors[n][2])/vectors[n][1];
			else
				vectors[n][3] = 1;
			vectors[n][8] = (vectors[n][4]+vectors[n][5]) /vectors[n][0];
			vectors[n][9] = (vectors[n][6]+vectors[n][7]) /vectors[n][0];
			vectors[n][10] = (vectors[n][4]+vectors[n][6]) /vectors[n][0];
			vectors[n][11] = (vectors[n][5]+vectors[n][7]) /vectors[n][0];
			vectors[n][4] = vectors[n][4] /vectors[n][0];
			vectors[n][5] = vectors[n][5] /vectors[n][0];
			vectors[n][6] = vectors[n][6] /vectors[n][0];
			vectors[n][7]= vectors[n][7] /vectors[n][0];
			vectors[n][0]= 1.0 /vectors[n][0];
			vectors[n][1]= 1.0 /vectors[n][1];
			vectors[n][2]= 1.0 /vectors[n][2];
		}
		
		//Fast Thinning Algorithm
		for(int n=0; n<length; n++){			
			boolean hasChanged = false;
			do{
				hasChanged = false;
				for(int i=1; i<x-1; i++){
					for(int j=1; j<y-1; j++){
						if(images[n][i][j]==1){
							int[] line = new int[9];
							line[0] = images[n][i-1][j];
							line[1] = images[n][i-1][j+1];
							line[2] = images[n][i][j+1];
							line[3] = images[n][i+1][j+1];
							line[4] = images[n][i+1][j];
							line[5] = images[n][i+1][j-1];
							line[6] = images[n][i][j-1];
							line[7] = images[n][i-1][j-1];
							int count=0;
							int trans = 0;
							for(int k=0; k<8; k++){
								count+=line[k];
								if(line[k]==0 && line[(k+1)%8]==1){
									trans++;
								}
							}
							if(2<=count && count<=6 && trans==1 && line[0]*line[2]*line[4]==0 && line[2]*line[4]*line[6]==0){
								temp[i][j] = 0;
								hasChanged = true;
							}else
								temp[i][j] = 1;
						}else{
							temp[i][j]=0;
						}
					}
				}
				for(int i=1; i<x-1; i++){
					for(int j=1; j<y-1; j++){
						images[n][i][j] = temp[i][j];
					}
				}
				for(int i=1; i<x-1; i++){
					for(int j=1; j<y-1; j++){
						if(images[n][i][j]==1){
							int[] line = new int[8];
							line[0] = images[n][i-1][j];
							line[1] = images[n][i-1][j+1];
							line[2] = images[n][i][j+1];
							line[3] = images[n][i+1][j+1];
							line[4] = images[n][i+1][j];
							line[5] = images[n][i+1][j-1];
							line[6] = images[n][i][j-1];
							line[7] = images[n][i-1][j-1];
							
							int count=0;
							int trans = 0;
							for(int k=0; k<8; k++){
								count+=line[k];
								if(line[k]==0 && line[(k+1)%8]==1){
									trans++;
								}
							}
							if(2<=count && count<=6 && trans==1 && line[0]*line[2]*line[6]==0 && line[0]*line[4]*line[6]==0){
								temp[i][j] = 0;
								hasChanged = true;
							}else
								temp[i][j] = 1;
						}else{
							temp[i][j]=0;
						}
					}
				}
				for(int i=1; i<x-1; i++){
					for(int j=1; j<y-1; j++){
						images[n][i][j] = temp[i][j];
					}
				}
			}while(hasChanged);
	
			vectors[n][12] = 0;
			vectors[n][13] = 0;
			vectors[n][14] = 0;
			vectors[n][15] = 0;
			vectors[n][16] = 0;
			for(int i=0; i<x; i++){
				for(int j=0; j<y; j++){
					if(j==(y/2) && temp[i][j]==1){
						vectors[n][12]+=1;
					}
					if(i==(x/2) && temp[i][j]==1){
						vectors[n][13]+=1;
					}
					int count = 0;
					for(int k=0; k<9 && i!=0 && j!=0 && i!=x-1 && j!=y-1; k++){
						count+=temp[i+(k%3)-1][j+(k/3)-1];
					}
					if(count==2 && temp[i][j]==1){
						vectors[n][14]+=1;
					}
					if(count==4 && temp[i][j]==1){
						vectors[n][15]+=1;
					}
					if(count==5 && temp[i][j]==1){
						vectors[n][16]+=1;
					}		
				}
			}
		}
	}
	
	//Constructor helper
	private int readInt(int off){
		return (data[off+3]&0xff) | ((data[off+2]&0xff)<<8) | ((data[off+1]&0xff)<<16) | ((data[off]&0xff)<<24);
	}	
	
	//Getters.
	public int getClassification(int n){
		return label[n+8]&0xff;
	}
	
	public int getLength(){
		return length;
	}
	
	public int getX(){
		return x;
	}
	
	public int getY(){
		return y;
	}	
	
	public static void main(String[] args) {
		//Must have all four files from MNIST database in the same folder to work.
		//Change this path to direct to the train-images.idx3-ubyte file.
		String trainingFileName = "C:\\Users\\bjnad\\Desktop\\Artificial Intelligence\\Project\\Hand Digit Recognition\\Hand Digit Recognition\\src\\MNIST Data\\train-images.idx3-ubyte";
		ClassifiedImageList trainingSet = new ClassifiedImageList(trainingFileName);
		
		//Change this path to direct to the t10k-images.idx3-ubyte file.
		String testFileName = "C:\\Users\\bjnad\\Desktop\\Artificial Intelligence\\Project\\Hand Digit Recognition\\Hand Digit Recognition\\src\\MNIST Data\\t10k-images.idx3-ubyte";
		ClassifiedImageList testingSet = new ClassifiedImageList(testFileName);
		
		//KNN k. Change this to change number of neighbors to use for classification.
		int k=8;
		
		int correct = 0;
		int[] correctClass = new int[10];
		int[] totalClass = new int[10];
		int[] countClass = new int[10];
		for(int i=0; i<10; i++){
			totalClass[i] = 0;
			correctClass[i] = 0;
			countClass[i]=0;
		}
		int[] kLocations = new int[k];
		int marker = 0;
		for(int i = 0; i<testingSet.getLength(); i++){
			for(int n=0; n<k; n++){
				kLocations[n] = 0;
			}
			int amount = 1;
			for(int j = 1; j<trainingSet.getLength(); j++){
				marker = j;
				for(int n=0; n<amount; n++){
					if(euclideanDistance(testingSet, trainingSet, i, kLocations[n]) > euclideanDistance(testingSet, trainingSet, i, marker)){
						int temp = kLocations[n];
						kLocations[n] = marker;
						marker = temp;
					}
				}
				if(amount <k)
					amount++;
			}
			totalClass[testingSet.getClassification(i)]++;
			
			for(int n=0; n<k; n++){
				countClass[trainingSet.getClassification(kLocations[n])]++;
			}
			int count=0;
			int loc = 0;
			for(int n=0; n<10; n++){
				if(count<countClass[n]){
					count = countClass[n];
					loc = n;
				}
				countClass[n]=0;
			}
			if(testingSet.getClassification(i) == loc){
				correct++;
				correctClass[testingSet.getClassification(i)]++;
			}
		}
		System.out.println("K: "+ k + " Correct: " + correct+" Total: "+testingSet.getLength()+ " Accuracy: "+ 1.0*correct/testingSet.getLength()*100+"%");
		for(int i=0; i<10; i++){
			System.out.println("Number: "+i+"\t"+correctClass[i] +"\t"+totalClass[i]+ "\t"+ 1.0*correctClass[i]/totalClass[i]*100+"% accuracy");
		}System.out.println();
		//Use this function to see individual images and their features.
		//printImage(testingSet, 0);
	}
	
	public static double euclideanDistance(ClassifiedImageList a, ClassifiedImageList b, int aLoc, int bLoc){
		double total = 0;
		for(int i=0; i<a.CLASSIFIERS; i++){
				total+= (a.vectors[aLoc][i]-b.vectors[bLoc][i])*(a.vectors[aLoc][i]-b.vectors[bLoc][i]);
		}
		return Math.sqrt(total);
	}
	
	//Use this method to print any image in any set along with its features and classification
	public static void printImage(ClassifiedImageList a, int n){
		for(int i=0; i<a.getX(); i++){
			for(int j=0; j<a.getY(); j++){
				System.out.print(a.images[n][i][j]);
			}System.out.println();
		}
		System.out.println("Classification: "+a.getClassification(n));
		System.out.println("Area: "+ a.vectors[n][0]);
		System.out.println("Height: "+a.vectors[n][1]);
		System.out.println("Width: "+a.vectors[n][2]);
		System.out.println("Width-Height Ratio: "+a.vectors[n][3]);
		System.out.println("Upper Left: "+a.vectors[n][4]);
		System.out.println("Upper Right: "+a.vectors[n][5]);
		System.out.println("Lower Left: "+a.vectors[n][6]);
		System.out.println("Lower Right: "+a.vectors[n][7]);
		System.out.println("Upper Area: "+a.vectors[n][8]);
		System.out.println("Lower Area: "+a.vectors[n][9]);
		System.out.println("Left Area: "+a.vectors[n][10]);
		System.out.println("Right Area: "+a.vectors[n][11]);
		System.out.println("Vertical Crossings: "+a.vectors[n][12]);
		System.out.println("Horizontal Crossigs: "+a.vectors[n][13]);
		System.out.println("Endpoints: "+a.vectors[n][14]);
		System.out.println("Branch Points: "+a.vectors[n][15]);
		System.out.println("Cross Poins: "+a.vectors[n][16]);
		System.out.println();
	}
}
