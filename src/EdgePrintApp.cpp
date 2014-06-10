#include "cinder/app/AppNative.h"
#include "cinder/gl/gl.h"
#include "cinder/gl/Texture.h"
#include "cinder/ImageIo.h"
#include "cinder/params/Params.h"
#include "CinderOpenCV.h"
#include <fstream>

using namespace ci;
using namespace ci::app;
using namespace std;

class EdgePrintApp : public AppNative {
public:
	void prepareSettings(Settings *pSettings);
	void setup();
	void setupGUI();
	void mouseDown( MouseEvent event );	
	void keyDown( KeyEvent pEvent );
	void update();
	void draw();
	
	void findFace();
	void cropFace();
	void updateFaceBox();
	void drawFaceBox();
	void drawContours();

	void exportContours();

	Surface8u SrcSurface, DstSurface;
	gl::Texture DstTexture;
	params::InterfaceGl GUI;

	double Thresh_0, Thresh_1;
	int PreBlurSize, PostBlurSize, SrcChannel, PadX, PadY, CenterX, CenterY;
	bool LGradient, DoCanny, DoPreBlur, DoPostBlur, UseGray, DoHist, UseROI, DoPreview, FindContours, DrawContours;

	Vec2f FaceCenter;
	cv::CascadeClassifier Face;
	vector<cv::Rect> FaceBoxes;
	vector<vector<cv::Point>> FaceContours;
	vector<vector<Vec2f>> FacePolygons;
	cv::Mat TempMat;
	Rectf FoundFaceBox, DrawFaceBox;
};

void EdgePrintApp::prepareSettings(Settings *pSettings)
{
	pSettings->setWindowSize(960,600);
}

void EdgePrintApp::setup()
{
	if(Face.load( getAssetPath("haarcascade_frontalface_alt.xml").string() ) )
		console() << "Loaded Haar Cascade." << endl;
	SrcSurface = loadImage(loadAsset("tree_crop.png"));
	DstSurface = Surface8u(SrcSurface.getWidth(), SrcSurface.getHeight(), false, SurfaceChannelOrder::RGB);

	setupGUI();
}

void EdgePrintApp::setupGUI()
{
	Thresh_0 = 40;
	Thresh_1 = 40;
	PadX=PadY=10;
	CenterX=CenterY=0;
	FaceCenter = Vec2f::zero();
	LGradient = false;
	DoCanny = false;
	DoPreBlur = DoPostBlur = false;
	PreBlurSize = PostBlurSize = 3;
	UseGray = false;
	SrcChannel = 1;
	DoHist = false;
	UseROI = false;
	DoPreview = false;
	FindContours = false;
	DrawContours = false;

	GUI = params::InterfaceGl("Canny Options", Vec2i(250,450), ColorA(0,0.5f,0,0.5f));
	GUI.setOptions("","position='20 20'");

	GUI.addSeparator();
	GUI.addButton("Find Face", bind(&EdgePrintApp::findFace, this));
	GUI.addParam("Face Margin Width", &PadX);
	GUI.addParam("Face Margin Height", &PadY);
	GUI.addParam("Face Center X", &CenterX);
	GUI.addParam("Face Center Y", &CenterY);
	GUI.addParam("Crop Face", &UseROI);

	GUI.addSeparator();
	GUI.addParam("Grayscale", &UseGray);
	GUI.addParam("Source Channel", &SrcChannel);
	GUI.addParam("Histogram Eq", &DoHist);
	GUI.addSeparator();
	GUI.addParam("Pre Blur", &DoPreBlur);
	GUI.addParam("Pre-Blur Size", &PreBlurSize);
	GUI.addSeparator();
	GUI.addParam("Threshold Min", &Thresh_0);
	GUI.addParam("Threshold Max", &Thresh_1);
	GUI.addParam("Use Gradient", &LGradient);
	GUI.addParam("Find Edges", &DoCanny);

	GUI.addSeparator();
	GUI.addParam("Post Blur", &DoPostBlur);
	GUI.addParam("Post-Blur Size", &PostBlurSize);
	GUI.addParam("Find Contours", &FindContours);
	GUI.addParam("Draw Contours", &DrawContours);

	GUI.addSeparator();
	GUI.addButton("Export Contours", bind(&EdgePrintApp::exportContours, this));
}

void EdgePrintApp::mouseDown( MouseEvent event )
{
}

void EdgePrintApp::keyDown( KeyEvent pEvent )
{
	if(pEvent.getCode()==KeyEvent::KEY_UP)
		--CenterY;
	else if(pEvent.getCode()==KeyEvent::KEY_DOWN)
		++CenterY;
	else if(pEvent.getCode()==KeyEvent::KEY_LEFT)
		--CenterX;
	else if(pEvent.getCode()==KeyEvent::KEY_RIGHT)
		++CenterX;

	else if(pEvent.getCode()==KeyEvent::KEY_w)
		++PadY;
	else if(pEvent.getCode()==KeyEvent::KEY_s)
		--PadY;
	else if(pEvent.getCode()==KeyEvent::KEY_a)
		++PadX;
	else if(pEvent.getCode()==KeyEvent::KEY_d)
		--PadX;
}

void EdgePrintApp::update()
{
	DstSurface = Surface(SrcSurface);
	if(UseROI)
	{
		updateFaceBox();
		cropFace();
	}
	if(UseGray)
		DstSurface = Surface(Channel(DstSurface));
	else
	{
		switch(SrcChannel)
		{
			case 1:
			{
				DstSurface = Surface(DstSurface.getChannelRed());
				break;
			}
			case 2:
			{
				DstSurface = Surface(DstSurface.getChannelGreen());
				break;
			}
			case 3:
			{
				DstSurface = Surface(DstSurface.getChannelBlue());
				break;
			}
		}
	}

	if(DoHist)
	{
		cv::Mat cTempEqMat;
		TempMat = toOcv(Channel(DstSurface));
		cv::equalizeHist(TempMat, cTempEqMat);
		DstSurface = fromOcv(cTempEqMat);
	}

	if(DoPreBlur)
	{
		TempMat = toOcv(DstSurface);
		cv::blur(TempMat, TempMat, cv::Size(PreBlurSize, PreBlurSize));
		DstSurface = fromOcv(TempMat);
	}

	if(DoCanny)
	{
		TempMat = toOcv(DstSurface);
		cv::Mat cCannyMat;
		cv::Canny(TempMat, cCannyMat, Thresh_0, Thresh_1, 3, LGradient);
		DstSurface = fromOcv(cCannyMat);
	}

	if(DoPostBlur)
	{
		TempMat = toOcv(DstSurface);
		cv::blur(TempMat, TempMat, cv::Size(PostBlurSize, PostBlurSize));
		DstSurface = fromOcv(TempMat);
	}

	if(FindContours)
	{
		FaceContours.clear();
		TempMat = toOcv(Channel(DstSurface));
		cv::findContours(TempMat, FaceContours,CV_RETR_TREE,CV_CHAIN_APPROX_SIMPLE, cv::Point(0,0));
	}
}

void EdgePrintApp::draw()
{
	// clear out the window with black
	gl::clear( Color( 0, 0, 0 ) ); 

	gl::color(Color::white());
	GUI.draw();

	gl::pushMatrices();
	gl::translate(300,20);
	gl::draw(SrcSurface, Vec2f::zero());

	if(FaceBoxes.size()>0)
		drawFaceBox();

	gl::translate(300,0);
	gl::color(Color::white());

	if(UseROI)
	{
		Vec2f cTL = Vec2f::zero();
		if(FaceBoxes.size()>0)
		{
			cTL = SrcSurface.getBounds().getCenter();
			cTL-=Vec2f(DrawFaceBox.getSize()*Vec2f(0.5f,0.5f));
		}
		gl::draw(DstSurface, cTL);
	}
	else
		gl::draw(DstSurface, Vec2f::zero());
	gl::color(Color(1,0.5f,0));
	gl::drawStrokedRect(SrcSurface.getBounds());

	gl::popMatrices();

	gl::pushMatrices();
	gl::scale(2,2,1);
	string cFps = "Current FPS: "+to_string(getFrameRate());
	gl::drawString(cFps, Vec2f(10,getWindowHeight()-320));
	gl::popMatrices();

	if(DrawContours&&FaceContours.size()>0)
		drawContours();
}

void EdgePrintApp::findFace()
{
	TempMat = toOcv(Channel(SrcSurface));
	FaceBoxes.clear();
	FoundFaceBox.set(-1,-1,-1,-1);
	Face.detectMultiScale(TempMat, FaceBoxes);
	if(FaceBoxes.size()>0)
	{
		FoundFaceBox = fromOcv(FaceBoxes[0]);
		updateFaceBox();
	}
}

void EdgePrintApp::cropFace()
{
	Surface8u cCropSurf(SrcSurface);
	if(FaceBoxes.size()>0)
	{
		cCropSurf = Surface8u(DrawFaceBox.getWidth(),DrawFaceBox.getHeight(),false,SurfaceChannelOrder::RGB);
		cCropSurf.copyFrom(SrcSurface, Area(DrawFaceBox), Vec2i(-DrawFaceBox.x1,-DrawFaceBox.y1));
	}

	DstSurface = Surface8u(cCropSurf);
}

void EdgePrintApp::updateFaceBox()
{
	DrawFaceBox = Rectf(FoundFaceBox);
	DrawFaceBox.x1-=PadX;
	DrawFaceBox.x2+=PadX;
	DrawFaceBox.y1-=PadY;
	DrawFaceBox.y2+=PadY;
	DrawFaceBox.offset(Vec2f(CenterX,CenterY));
}

void EdgePrintApp::drawFaceBox()
{
	gl::color(Color(0,1,0));
	gl::drawStrokedRect(DrawFaceBox);	
}

void EdgePrintApp::drawContours()
{
	gl::color( Color(1,1,0));
	gl::pushMatrices();
	gl::translate(600,20);
	if(UseROI&&FaceBoxes.size()>0)
	{
		Vec2f cTL = SrcSurface.getBounds().getCenter();
		cTL-=Vec2f(DrawFaceBox.getSize()*Vec2f(0.5f,0.5f));
		gl::translate(cTL);
	}

	for(auto cit=FaceContours.begin();cit!=FaceContours.end();++cit)
	{
		vector<cv::Point> cContour = *cit;
		for(uint16_t pi=1;pi<cContour.size();++pi)
			gl::drawLine(fromOcv(cContour[pi-1]), fromOcv(cContour[pi]));
		gl::drawLine(fromOcv(cContour[cContour.size()-1]), fromOcv(cContour[0]));
	}
	gl::popMatrices();
}

void EdgePrintApp::exportContours()
{
	if(FaceContours.size()>0)
	{
		string cJsonStr = "{";
		for(uint16_t ci=0;ci<FaceContours.size();++ci)
		{
			string cNodeStr = "\"contour_"+to_string(ci)+"\":[";
			vector<cv::Point> cContour = FaceContours[ci];
			vector<cv::Point> cPolygon;
			cv::approxPolyDP(cContour, cPolygon,0,true);
			for(uint16_t pi=0;pi<cPolygon.size();++pi)
			{
				int cX = cPolygon[pi].x;
				int cY = cPolygon[pi].y;
				cNodeStr+="{\"x\":"+to_string(cX)+",\"y\":"+to_string(cY)+"}";
				if(pi<cPolygon.size()-1)
					cNodeStr+=",";
			}
			cNodeStr+="]";
			if(ci<FaceContours.size()-1)
				cNodeStr+=",";
			cJsonStr+=cNodeStr;
		}
		cJsonStr+="}";

		ofstream cFile;
		cFile.open("contours.txt");
		cFile << cJsonStr;
		cFile.close();
	}
}

CINDER_APP_NATIVE( EdgePrintApp, RendererGl )
