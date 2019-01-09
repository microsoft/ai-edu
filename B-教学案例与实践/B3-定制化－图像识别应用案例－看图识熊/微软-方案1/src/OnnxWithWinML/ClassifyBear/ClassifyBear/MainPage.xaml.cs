// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices.WindowsRuntime;
using System.Threading.Tasks;
using Windows.AI.MachineLearning;
using Windows.Foundation;
using Windows.Foundation.Collections;
using Windows.Graphics.Display;
using Windows.Graphics.Imaging;
using Windows.Media;
using Windows.Storage;
using Windows.Storage.Pickers;
using Windows.Storage.Streams;
using Windows.UI.Xaml;
using Windows.UI.Xaml.Controls;
using Windows.UI.Xaml.Controls.Primitives;
using Windows.UI.Xaml.Data;
using Windows.UI.Xaml.Input;
using Windows.UI.Xaml.Media;
using Windows.UI.Xaml.Media.Imaging;
using Windows.UI.Xaml.Navigation;

// https://go.microsoft.com/fwlink/?LinkId=402352&clcid=0x804 上介绍了“空白页”项模板

namespace ClassifyBear
{
    /// <summary>
    /// 可用于自身或导航至 Frame 内部的空白页。
    /// </summary>
    public sealed partial class MainPage : Page
    {
        public MainPage()
        {
            this.InitializeComponent();
        }

        private void TbRun_Tapped(object sender, TappedRoutedEventArgs e)
        {
            tbBearType.Text = string.Empty;

            Uri imageUri = null;
            try
            {
                imageUri = new Uri(tbImageUrl.Text);
            }
            catch (Exception)
            {
                tbBearType.Text = "URL不合法";
                return;
            }

            tbBearType.Text = "加载图片...";

            imgBear.Source = new BitmapImage(imageUri);
        }

        private void ImgBear_ImageOpened(object sender, RoutedEventArgs e)
        {
            RecognizeBear();
        }

        private void ImgBear_ImageFailed(object sender, ExceptionRoutedEventArgs e)
        {
            tbBearType.Text = "图片加载失败";
        }

        private async Task<BearModelInput> GetInputData()
        {
            // 将图片控件重绘到图片上
            RenderTargetBitmap rtb = new RenderTargetBitmap();
            await rtb.RenderAsync(imgBear);

            // 取得所有像素值
            var pixelBuffer = await rtb.GetPixelsAsync();

            // 构造模型需要的输入格式
            SoftwareBitmap softwareBitmap = SoftwareBitmap.CreateCopyFromBuffer(pixelBuffer, BitmapPixelFormat.Bgra8, rtb.PixelWidth, rtb.PixelHeight);
            VideoFrame videoFrame = VideoFrame.CreateWithSoftwareBitmap(softwareBitmap);
            ImageFeatureValue imageFeatureValue = ImageFeatureValue.CreateFromVideoFrame(videoFrame);

            BearModelInput bearModelInput = new BearModelInput();
            bearModelInput.data = imageFeatureValue;
            return bearModelInput;
        }

        private async void RecognizeBear()
        {
            // 加载模型
            StorageFile modelFile = await StorageFile.GetFileFromApplicationUriAsync(new Uri($"ms-appx:///Assets/BearModel.onnx"));
            BearModelModel model = await BearModelModel.CreateFromStreamAsync(modelFile);

            // 构建输入数据
            BearModelInput bearModelInput = await GetInputData();

            // 推理
            BearModelOutput output = await model.EvaluateAsync(bearModelInput);

            tbBearType.Text = output.classLabel.GetAsVectorView().ToList().FirstOrDefault();
        }
    }
}
