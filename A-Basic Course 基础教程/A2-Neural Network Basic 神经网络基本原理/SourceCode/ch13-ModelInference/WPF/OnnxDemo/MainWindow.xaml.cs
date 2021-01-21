// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Ink;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace OnnxDemo
{
    /// <summary>
    /// MainWindow.xaml 的交互逻辑
    /// </summary>
    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();

            InitInk();
        }

        private void InitInk()
        {
            // 将画笔改为白色
            var attr = new DrawingAttributes();
            attr.Color = Colors.White;
            attr.IgnorePressure = true;
            attr.StylusTip = StylusTip.Ellipse;
            attr.Height = 24;
            attr.Width = 24;
            inkCanvas.DefaultDrawingAttributes = attr;

            // 每次画完一笔时，都触发此事件进行识别
            inkCanvas.StrokeCollected += InkCanvas_StrokeCollected;
        }

        private void InkCanvas_StrokeCollected(object sender, InkCanvasStrokeCollectedEventArgs e)
        {
            // 从画布中进行识别
            RecogNumberFromInk();
        }

        private void BtnClean_Click(object sender, RoutedEventArgs e)
        {
            // 清除画布
            inkCanvas.Strokes.Clear();
            lbResult.Text = string.Empty;
        }

        private BitmapSource RenderToBitmap(FrameworkElement canvas, int scaledWidth, int scaledHeight)
        {
            // 将画布渲染到bitmap上
            RenderTargetBitmap rtb = new RenderTargetBitmap((int)canvas.Width, (int)canvas.Height, 96d, 96d, PixelFormats.Default);
            rtb.Render(canvas);

            // 调整bitmap的大小为28*28，与模型的输入保持一致
            TransformedBitmap tfb = new TransformedBitmap(rtb, new ScaleTransform(scaledWidth / rtb.Width, scaledHeight / rtb.Height));
            return tfb;
        }

        public byte[] GetPixels(BitmapSource source)
        {
            if (source.Format != PixelFormats.Bgra32)
                source = new FormatConvertedBitmap(source, PixelFormats.Bgra32, null, 0);

            int width = source.PixelWidth;
            int height = source.PixelHeight;
            byte[] data = new byte[width * 4 * height];

            source.CopyPixels(data, width * 4, 0);
            return data;
        }

        public float[] GetInputDataFromInk()
        {
            var bitmap = RenderToBitmap(inkCanvas, 28, 28);
            var imageBytes = GetPixels(bitmap);

            float[] data = new float[784];
            for (int i = 0; i < 784; i++)
            {
                // 画布为黑白色的，可以直接取RGB中的一个分量作为此像素的色值
                int baseIndex = 4 * i;
                data[i] = imageBytes[baseIndex];
            }

            return data;
        }

        private void RecogNumberFromInk()
        {
            // 从画布得到输入数组
            var inputData = GetInputDataFromInk();

            // 从文件中加载模型
            string modelPath = AppDomain.CurrentDomain.BaseDirectory + "mnist.onnx";

            using (var session = new InferenceSession(modelPath))
            {
                // 支持多个输入，对于mnist模型，只需要一个输入
                var container = new List<NamedOnnxValue>();

                // 输入是大小1*784的一维数组
                var tensor = new DenseTensor<float>(inputData, new int[] { 1, 784 });

                // 输入的名称是port
                container.Add(NamedOnnxValue.CreateFromTensor<float>("fc1x", tensor));

                // 推理
                var results = session.Run(container);

                // 输出结果是IReadOnlyList<NamedOnnxValue>，支持多个输出，对于mnist模型，只有一个输出
                var result = results.FirstOrDefault()?.AsTensor<float>()?.ToList();

                // 从输出中取出得分最高的
                var max = result.IndexOf(result.Max());

                // 显示在控件中
                lbResult.Text = max.ToString();
            }
        }
    }
}
