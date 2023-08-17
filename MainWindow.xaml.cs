using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using MathNet.Numerics.Statistics;
using System.Windows.Media.Imaging;
using System.IO;
using System.Threading;
using System.Windows.Threading;
using System.Reflection;

namespace LightGlue
{
    /// <summary>
    /// MainWindow.xaml 的交互逻辑
    /// </summary>
    public partial class MainWindow : Window
    {
        List<string> mImgList;
        /// <summary>
        /// ONNX Inference Session
        /// </summary>
        InferenceSession mSuperPoint;
        InferenceSession mLightGlue;
        /// <summary>
        /// 存储SuperPoint提取的特征点对应的描述符
        /// </summary>
        Dictionary<string, Tensor<float>> mFeatureDes=new Dictionary<string, Tensor<float>>();
        /// <summary>
        /// 存储SuperPoint提取的特征点
        /// </summary>
        Dictionary<string, Tensor<long>> mFeatureKeyPoints=new Dictionary<string, Tensor<long>>();
        /// <summary>
        /// 存储SuperPoint提取的特征点对应的置信分数
        /// </summary>
        Dictionary<string, Tensor<float>> mFeatureScores=new Dictionary<string, Tensor<float>>();
        /// <summary>
        /// 存储原始图像数据
        /// </summary>
        Dictionary<string, float[]> mImgDatas=new Dictionary<string, float[]>();
        /// <summary>
        /// 存储图像宽高
        /// </summary>
        Dictionary<string, int> mWidth=new Dictionary<string, int>();
        Dictionary<string, int> mHeight=new Dictionary<string, int>();
        /// <summary>
        /// 存储图像水平方向的偏移
        /// </summary>
        Dictionary<string, int> mXOffset = new Dictionary<string, int>();
        /// <summary>
        /// 得分前N个关键带你
        /// </summary>
        Dictionary<string, int> mTopNKeyPoints = new Dictionary<string, int>();
        /// <summary>
        /// 图像控件宽高
        /// </summary>
        double mControlWid;
        double mControlHei;
        Dispatcher UI;
        public MainWindow()
        {
            InitializeComponent();
            this.InitUI();
            this.LoadModel();

            this.mImgList = new List<string>();
            this.UI = Dispatcher.CurrentDispatcher;
            this.SizeChanged += MainWindow_SizeChanged;
        }

        private void MainWindow_SizeChanged(object sender, SizeChangedEventArgs e)
        {
            this.InitUI();
        }

        /// <summary>
        /// 初始化图像控件
        /// </summary>
        void InitUI()
        {
            this.mImage.Width = this.Width;
            this.mImage.Height = this.Height * 0.8f;
            this.mImgCanvas.Width = this.Width;
            this.mImgCanvas.Height = this.Height * 0.8f;
            this.mControlWid = this.mImage.Width;
            this.mControlHei = this.mImage.Height;          
        }
        /// <summary>
        /// Load ONNX format Model
        /// </summary>
        void LoadModel()
        {
            string exepath = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location);
            this.mSuperPoint = new InferenceSession(exepath+"\\superpoint.onnx");
            this.mLightGlue = new InferenceSession(exepath+"\\superpoint_lightglue.onnx");
        }
        /// <summary>
        /// 特征提取
        /// feature extraction
        /// </summary>
        void Feature()
        {
            this.mFeatureKeyPoints.Clear();
            this.mFeatureDes.Clear();
            this.mTopNKeyPoints.Clear();

            this.UpdateUI(UIType.IMAGE,"Featureing ");

            Transforms tranform = new Transforms();
            foreach (var imgp in this.mImgList)
            {
                float[] imgdata = tranform.ApplyImage(this.mImgDatas[imgp]);
                var inputTensor= new DenseTensor<float>(imgdata, new[] { 1, 1, this.mHeight[imgp],this.mWidth[imgp]});
                // Create a dictionary to specify named inputs
                var inputs = new List<NamedOnnxValue>
                {
                    NamedOnnxValue.CreateFromTensor("image", inputTensor)
                };
                // Run the inference
                var results = this.mSuperPoint.Run(inputs);
                // Get the output tensors
                this.mFeatureKeyPoints[imgp] = results.First(o => o.Name == "keypoints").AsTensor<long>();
                this.mFeatureScores[imgp] = results.First(o => o.Name == "scores").AsTensor<float>();
                this.mFeatureDes[imgp] = results.First(o => o.Name == "descriptors").AsTensor<float>();
            }
            UI.Invoke(new Action(delegate
            {
                this.UpdateUI(UIType.KEYPOINT,"Show top ");
                this.DrawCircle();
            }));
           
        }
        /// <summary>
        /// Show Keypoints with circles，The higher the score, the darker the color
        /// </summary>
        void DrawCircle()
        {          
            int stiImgWidth = 0;
            int stiImgHeight = 0;
            this.GetStiImgSize(ref stiImgWidth, ref stiImgHeight);
            this.ClearAllAnation();

            foreach (var img in this.mImgList)
            {             
                var scores = this.GetTopNScores(img);
                var keypoints = this.GetTopNKeyPoints(img);

                for (int i= 0;i<keypoints.Count/2;i++)
                {
                    int x = (int)keypoints[2 * i];
                    int y = (int)keypoints[2 * i + 1];
                    double xratio = this.mControlWid / stiImgWidth;//图像坐标转控件坐标
                    double yratio = this.mControlHei / stiImgHeight;
                    double ImageX = xratio * (this.mXOffset[img] + x);
                    double ImageY = yratio * y;
                    System.Windows.Point imagePoint = new System.Windows.Point(ImageX, ImageY); // 假设要转换的图像坐标为(100, 100)
                    System.Windows.Point canvasPoint = this.mImage.TransformToAncestor(this.mImgCanvas).Transform(imagePoint);
                    System.Windows.Shapes.Ellipse ellipse = new System.Windows.Shapes.Ellipse();
                    ellipse.Width = 5;//圆的半径
                    ellipse.Height = 5;
                    float r = 255 * scores[i] * 100;
                    if (r > 255)
                        r = 255;
                    float g = 255 * scores[i] * 100;
                    if (g > 255)
                        g = 255;
                    ellipse.Fill = new System.Windows.Media.SolidColorBrush(System.Windows.Media.Color.FromRgb((byte)r, (byte)g, 0));
                    System.Windows.Controls.Canvas.SetLeft(ellipse, canvasPoint.X);
                    System.Windows.Controls.Canvas.SetTop(ellipse, canvasPoint.Y);
                    this.mImgCanvas.Children.Add(ellipse);
                }       
            }        
        }
        /// <summary>
        /// 动态显示关键点信息
        /// </summary>
        void UpdateUI(UIType type,string msg)
        {
            this.mDynamicUI.Children.Clear();
            this.mDynamicUI.ColumnDefinitions.Clear();

            int stiImgWidth = 0;
            int stiImgHeight = 0;
            this.GetStiImgSize(ref stiImgWidth, ref stiImgHeight);
            double xratio = this.mControlWid / stiImgWidth;

            foreach (var v in this.mImgList)
            {
                double imgControlWid = this.mWidth[v] * xratio;//每个图像所占图像控件宽度

                System.Windows.Controls.ColumnDefinition newColumn = new System.Windows.Controls.ColumnDefinition();
                newColumn.Width = new GridLength(imgControlWid);
                this.mDynamicUI.ColumnDefinitions.Add(newColumn);
            }

            foreach (var v in this.mImgList)
            {
                double imgControlWid = this.mWidth[v] * xratio;//每个图像所占图像控件宽度
                System.Windows.Controls.DockPanel dc = new System.Windows.Controls.DockPanel();
                dc.Width = imgControlWid;
                System.Windows.Controls.Grid.SetColumn(dc,this.mImgList.IndexOf(v));
                
                if (type == UIType.KEYPOINT)
                {
                    System.Windows.Controls.TextBlock texpre = new System.Windows.Controls.TextBlock();
                    texpre.Text = msg;
                    texpre.VerticalAlignment = VerticalAlignment.Center;
                    texpre.Margin = new Thickness(5, 0, 0, 0);
                    dc.Children.Add(texpre);

                    System.Windows.Controls.TextBox tb = new System.Windows.Controls.TextBox();
                    tb.Name = this.GetInvalidPathName(v);
                    tb.Width = 50;
                    tb.Height = 20;
                    tb.Text = this.mFeatureKeyPoints[v].Count().ToString();
                    tb.VerticalAlignment = VerticalAlignment.Center;
                    tb.KeyUp += Tb_KeyUp;
                    tb.Margin = new Thickness(5, 0, 0, 0);
                    dc.Children.Add(tb);

                    System.Windows.Controls.TextBlock texafter = new System.Windows.Controls.TextBlock();
                    texafter.Text = string.Format("of total {0} keypoints", this.mFeatureKeyPoints[v].Count());
                    texafter.VerticalAlignment = VerticalAlignment.Center;
                    texafter.Margin = new Thickness(5, 0, 0, 0);
                    dc.Children.Add(texafter);
                }
                else if (type == UIType.IMAGE)//Load Image
                {
                    System.Windows.Controls.TextBlock texpre = new System.Windows.Controls.TextBlock();
                    texpre.Text = msg;
                    texpre.VerticalAlignment = VerticalAlignment.Center;
                    texpre.Margin = new Thickness(5, 0, 0, 0);
                    dc.Children.Add(texpre);
                }
                                 
                this.mDynamicUI.Children.Add(dc);
            }
          
        }

        private void Tb_KeyUp(object sender, System.Windows.Input.KeyEventArgs e)
        {
            if (e.Key != System.Windows.Input.Key.Enter)
                return;

            System.Windows.Controls.TextBox tb = sender as System.Windows.Controls.TextBox;
            if (tb == null)
                return;


            var cur = this.mImgList.Find(f => { return this.GetInvalidPathName(f) == tb.Name; });
            this.mTopNKeyPoints[cur] = int.Parse(tb.Text);
           
            this.DrawCircle();
        }
        List<float> GetTopNScores(string img)
        {
            var scores = this.mFeatureScores[img].ToList();//得分

            int TopN = 0;
            if (this.mTopNKeyPoints.ContainsKey(img))
                TopN = this.mTopNKeyPoints[img];
            else
                TopN = scores.Count();
            var topNScores = scores.OrderByDescending(s => s).Take(TopN).ToList();

            return topNScores;
        }
        /// <summary>
        /// 根据用户设置获取得分最高的关键点
        /// </summary>
        /// <returns></returns>
        List<long> GetTopNKeyPoints(string img)
        {
            var scores = this.mFeatureScores[img].ToList();//得分
            var topNScores = this.GetTopNScores(img);
            var keypoints = this.mFeatureKeyPoints[img].ToList();//关键点

            List<long> kps = new List<long>();
            foreach (var s in topNScores)
            {
                int index = scores.IndexOf(s);
                kps.Add(keypoints[2 * index]);
                kps.Add(keypoints[2 * index + 1]);
            }
         
            return kps;
        }
        string GetInvalidPathName(string path)
        {
            char[] invalidChars = Path.GetInvalidFileNameChars();
            string imgname = path.Clone() as string;
            foreach (char c in invalidChars)
            {
                imgname = imgname.Replace(c.ToString(), "");
                imgname = imgname.Replace(".", "");
            }
            return imgname;
        }

        /// <summary>
        /// 特征点匹配
        /// </summary>
        void Match()
        {
            this.ClearLineAnation();
            for (int k = 0; k < this.mImgList.Count(); k++)
            {
                for (int i = k+1; i < this.mImgList.Count(); i++)
                {
                    string imgpath1 = this.mImgList[k];
                    var keypoints1 = this.mFeatureKeyPoints[imgpath1].ToList();
                    float[] k1 = new float[keypoints1.Count];
                    for (int j = 0; j < keypoints1.Count / 2; j++)
                    {
                        k1[2 * j] = (keypoints1[2 * j] - this.mWidth[imgpath1]) / (float)this.mWidth[imgpath1];
                        k1[2 * j + 1] = (keypoints1[2 * j + 1] - this.mHeight[imgpath1]) / (float)this.mHeight[imgpath1];
                    }
                    var kp1t = new DenseTensor<float>(k1, this.mFeatureKeyPoints[imgpath1].Dimensions.ToArray());

                    string imgpath2 = this.mImgList[i];
                    var keypoints2 = this.mFeatureKeyPoints[imgpath2].ToList();
                    float[] k2 = new float[keypoints2.Count];
                    for (int j = 0; j < keypoints2.Count / 2; j++)
                    {
                        k2[2 * j] = (keypoints2[2 * j] - this.mWidth[imgpath2]) / (float)this.mWidth[imgpath2];
                        k2[2 * j + 1] = (keypoints2[2 * j + 1] - this.mHeight[imgpath2]) / (float)this.mHeight[imgpath2];
                    }
                    var kp2t = new DenseTensor<float>(k2, this.mFeatureKeyPoints[imgpath2].Dimensions.ToArray());

                    var inputs3 = new List<NamedOnnxValue>
                    {
                        NamedOnnxValue.CreateFromTensor("kpts0", kp1t),
                        NamedOnnxValue.CreateFromTensor("kpts1", kp2t),

                        NamedOnnxValue.CreateFromTensor("desc0", this.mFeatureDes[imgpath1]),
                        NamedOnnxValue.CreateFromTensor("desc1", this.mFeatureDes[imgpath2])
                    };

                    var results = this.mLightGlue.Run(inputs3);
                    var match0 = results.First(o => o.Name == "matches0").AsTensor<Int64>();
                    var matchScore0 = results.First(o => o.Name == "mscores0").AsTensor<float>().ToList();
                    var match1 = results.First(o => o.Name == "matches1").AsTensor<Int64>();

                    List<Int64> mt1 = new List<Int64>();
                    List<Int64> mt2 = new List<Int64>();
                    var match0List = match0.ToList();
                    var match1List = match1.ToList();

                    for (int j = 0; j < match0List.Count; j++)
                    {
                        if (match0List[j] > -1 && matchScore0[j]>0.0f && match1List[(int)match0List[j]] == j)
                        {
                            mt1.Add(j);
                            mt2.Add(match0List[j]);
                        }
                    }

                    List<long> kp1 = new List<long>();
                    List<long> kp2 = new List<long>();


                    var topNScores1 = this.GetTopNScores(imgpath1);
                    var topNScores2 = this.GetTopNScores(imgpath2);
                    var scores1 = this.mFeatureScores[imgpath1].ToList();//得分
                    var scores2 = this.mFeatureScores[imgpath2].ToList();//得分


                    for (int j = 0; j < mt1.Count(); j++)
                    {
                        int inex1 = (int)mt1[j];
                        int inex2 = (int)mt2[j];

                        var t1 = topNScores1.Find(e => scores1.IndexOf(e) == inex1);
                        var t2 = topNScores2.Find(e => scores2.IndexOf(e) == inex2);
                        if (t1 != 0 && t2 != 0)
                        {
                            int x1 = (int)keypoints1[2 * inex1] + this.mXOffset[imgpath1];
                            int y1 = (int)keypoints1[2 * inex1 + 1];
                            kp1.Add(x1);
                            kp1.Add(y1);

                            int x2 = (int)keypoints2[2 * inex2] + this.mXOffset[imgpath2];
                            int y2 = (int)keypoints2[2 * inex2 + 1];
                            kp2.Add(x2);
                            kp2.Add(y2);

                        }
                    }
                    this.DrawLine(kp1, kp2);
                }
            }                                
        }
        /// <summary>
        /// Draw KeyPoint Match lines
        /// </summary>
        /// <param name="kp1"></param>
        /// <param name="kp2"></param>
        void DrawLine(List<long>kp1,List<long>kp2)
        {         
            int stiImgWidth = 0;
            int stiImgHeight = 0;
            this.GetStiImgSize(ref stiImgWidth, ref stiImgHeight);
            double xratio = this.mControlWid / stiImgWidth;
            double yratio = this.mControlHei / stiImgHeight;

            for (int j = 0; j < kp1.Count()/2; j++)
            {             
                System.Windows.Shapes.Line line = new System.Windows.Shapes.Line();
                line.StrokeThickness = 2;
                int x1 = (int)kp1[2 * j];
                int y1 = (int)kp1[2 * j + 1];

                int x2 = (int)kp2[2 * j];
                int y2 = (int)kp2[2 * j + 1];

                double ImageX1 = xratio * x1;
                double ImageY1 = yratio * y1;

                double ImageX2 = xratio * x2;
                double ImageY2 = yratio * y2;

                System.Windows.Point imagePoint1 = new System.Windows.Point(ImageX1, ImageY1); // 假设要转换的图像坐标为(100, 100)
                System.Windows.Point canvasPoint1 = this.mImage.TransformToAncestor(this.mImgCanvas).Transform(imagePoint1);
                line.Stroke = new System.Windows.Media.SolidColorBrush(System.Windows.Media.Color.FromRgb(255, 0, 0));
                line.X1 = canvasPoint1.X;
                line.Y1 = canvasPoint1.Y;
                System.Windows.Point imagePoint2 = new System.Windows.Point(ImageX2, ImageY2); // 假设要转换的图像坐标为(100, 100)
                System.Windows.Point canvasPoint2 = this.mImage.TransformToAncestor(this.mImgCanvas).Transform(imagePoint2);
                line.X2 = canvasPoint2.X;
                line.Y2 = canvasPoint2.Y;
                this.mImgCanvas.Children.Add(line);
            }
        }
        void ClearLineAnation()
        {
            List<System.Windows.Shapes.Shape> todel = new List<System.Windows.Shapes.Shape>();

            foreach (var v in this.mImgCanvas.Children)
            {
                if (v is System.Windows.Shapes.Line)
                    todel.Add(v as System.Windows.Shapes.Line);

            }
            todel.ForEach(e => { this.mImgCanvas.Children.Remove(e); });
        }
        /// <summary>
        /// 清空圆和直线标注
        /// </summary>
        void ClearAllAnation()
        {
            List<System.Windows.Shapes.Shape> todel = new List<System.Windows.Shapes.Shape>();

            foreach (var v in this.mImgCanvas.Children)
            {
                if (v is System.Windows.Shapes.Shape)
                    todel.Add(v as System.Windows.Shapes.Shape);

            }
            todel.ForEach(e => { this.mImgCanvas.Children.Remove(e); });
        }
        /// <summary>
        /// 将多福图像水平拼接在一起，获取拼接图像的宽高
        /// </summary>
        /// <param name="wid"></param>
        /// <param name="hei"></param>
        void GetStiImgSize(ref int wid,ref int hei)
        {
            foreach (var img in this.mImgList)
            {
                wid += this.mWidth[img];
                if (hei < this.mHeight[img])
                    hei = this.mHeight[img];
            }
        }
        /// <summary>
        /// 图像路径选择
        /// </summary>
        private void SelectFileButton_Click(object sender, RoutedEventArgs e)
        {
            // Create OpenFileDialog
            Microsoft.Win32.OpenFileDialog openFileDialog = new Microsoft.Win32.OpenFileDialog();
            openFileDialog.Multiselect = true; // 设置允许多选文件
            // Set filter for file extension and default file extension
            openFileDialog.DefaultExt = ".png";
            openFileDialog.Filter = "Image files (*.jpg, *.jpeg, *.png)|*.jpg;*.jpeg;*.png|All files (*.*)|*.*";

            // Display OpenFileDialog by calling ShowDialog method
            Nullable<bool> result = openFileDialog.ShowDialog();

            // Get the selected file name and display in a TextBox
            if (result == true)
            {
                this.mImgList.Clear();
                this.mImgList.AddRange(openFileDialog.FileNames);
                this.LoadImage();
            }
        }
        /// <summary>
        /// 加载图像
        /// </summary>
        void LoadImage()
        {
            this.ClearAllAnation();

            Dictionary<string, byte[]> ColorData = new Dictionary<string, byte[]>();
            foreach (var img in this.mImgList)
            {               
                Bitmap image = new Bitmap(img);
                int width = image.Width;
                int height = image.Height;

                byte[] pixelbs = new byte[4*width * height];
                float[] pixels = new float[width * height];
                for (int y = 0; y < height; y++)
                {
                    for (int x = 0; x < width; x++)
                    {
                        Color color = image.GetPixel(x, y);
                        byte gray = (byte)((color.R + color.G + color.B) / 3); // 计算灰度值
                        int index = x + y * width;
                        pixels[index] = gray;

                        int ind = y * width + x;
                        pixelbs[4 * ind] = color.B;  // Blue
                        pixelbs[4 * ind + 1] = color.G;  // Green
                        pixelbs[4 * ind + 2] = color.R;  // Red
                        pixelbs[4 * ind + 3] = 255;  // Alpha
                    }
                }
                ColorData[img] = pixelbs;
                this.mImgDatas[img] = pixels;             
                this.mWidth[img] = width;
                this.mHeight[img] = height;
            }

            int stiImgWidth = 0;
            int stiImgHeight = 0;
            this.GetStiImgSize(ref stiImgWidth,ref stiImgHeight);
            WriteableBitmap bp = new WriteableBitmap(stiImgWidth,stiImgHeight, 96, 96, System.Windows.Media.PixelFormats.Pbgra32, null);          
            int widthoffset = 0;
            foreach (var img in this.mImgList)
            {
                //byte[] imgb = new byte[this.mImgDatas[img].Count()];
                //for (int i=0;i< imgb.Count();i++)
                //{
                //    imgb[i] = (byte)this.mImgDatas[img][i];
                //}
                this.mXOffset[img] = widthoffset;
                bp.WritePixels(new Int32Rect(widthoffset, 0, this.mWidth[img], this.mHeight[img]), ColorData[img], this.mWidth[img]*4, 0);
                widthoffset += this.mWidth[img];
            }
            UI.BeginInvoke(new Action(delegate
            {
                // 创建一个BitmapImage对象，将WriteableBitmap作为源
                this.mImage.Source = bp;
                this.UpdateUI(UIType.IMAGE, "Loading Image Finished");
            }));
          
        }
        private void ImgMatch_Click(object sender, RoutedEventArgs e)
        {
            this.Match();
        }

        private void ImgFeature_Click(object sender, RoutedEventArgs e)
        {
            this.Feature();
        }


    }

    class Transforms
    {
        public Transforms()
        {
           
        }
        /// <summary>
        /// 变换图像，将原始图像变换大小
        /// </summary>
        /// <returns></returns>
        public float[] ApplyImage(float[] bp)
        {
            ////计算均值
            //float mean = (float)MathNet.Numerics.Statistics.Statistics.Mean(bp);
            ////计算标准差
            //float stdDev = (float)MathNet.Numerics.Statistics.Statistics.StandardDeviation(bp);

            //for (int i = 0; i < bp.Count(); i++)
            //{
            //    bp[i] = (bp[i] - mean) / stdDev;
            //}
            //return bp;

            for (int i = 0; i < bp.Count(); i++)
            {
                bp[i] = bp[i]/255.0f;
            }
            return bp;
        }
   
    }
    public enum UIType
    {
        IMAGE,
        KEYPOINT
    }
}
