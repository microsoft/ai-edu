// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices.WindowsRuntime;
using System.Threading.Tasks;
using Windows.AI.MachineLearning;
using Windows.Foundation;
using Windows.Foundation.Collections;
using Windows.Storage;
using Windows.UI;
using Windows.UI.Core;
using Windows.UI.Input.Inking;
using Windows.UI.Xaml;
using Windows.UI.Xaml.Controls;
using Windows.UI.Xaml.Controls.Primitives;
using Windows.UI.Xaml.Data;
using Windows.UI.Xaml.Input;
using Windows.UI.Xaml.Media;
using Windows.UI.Xaml.Media.Imaging;
using Windows.UI.Xaml.Navigation;

// https://go.microsoft.com/fwlink/?LinkId=402352&clcid=0x804 上介绍了“空白页”项模板

namespace OnnxDemo
{
    /// <summary>
    /// 可用于自身或导航至 Frame 内部的空白页。
    /// </summary>
    public sealed partial class MainPage : Page
    {
        public MainPage()
        {
            this.InitializeComponent();

            InitInk();
        }

        private void InitInk()
        {
            inkCanvas.InkPresenter.InputDeviceTypes = CoreInputDeviceTypes.Mouse | CoreInputDeviceTypes.Touch;
            var attr = new InkDrawingAttributes();
            attr.Color = Colors.White;
            attr.IgnorePressure = true;
            attr.PenTip = PenTipShape.Circle;
            attr.Size = new Size(24, 24);
            inkCanvas.InkPresenter.UpdateDefaultDrawingAttributes(attr);

            inkCanvas.InkPresenter.StrokesCollected += InkPresenter_StrokesCollected;
        }

        private void InkPresenter_StrokesCollected(InkPresenter sender, InkStrokesCollectedEventArgs args)
        {
            RecogNumberFromInk();
        }

        private void btnClean_Tapped(object sender, TappedRoutedEventArgs e)
        {
            inkCanvas.InkPresenter.StrokeContainer.Clear();
            lbResult.Text = string.Empty;
        }

        public async Task<float[]> GetInputDataFromInk()
        {
            // 将画布渲染到大小为28*28的bitmap上，与模型的输入保持一致
            RenderTargetBitmap renderBitmap = new RenderTargetBitmap();
            await renderBitmap.RenderAsync(inkGrid, 28, 28);

            // 取出所有像素点的色值
            var buffer = await renderBitmap.GetPixelsAsync();
            var imageBytes = buffer.ToArray();

            float[] data = new float[784];
            for (int i = 0; i < 784; i++)
            {
                // 画布为黑白色的，可以直接取RGB中的一个分量作为此像素的色值
                int baseIndex = 4 * i;
                data[i] = imageBytes[baseIndex];
            }

            return data;
        }

        private async void RecogNumberFromInk()
        {
            // 从文件加载模型
            var modelFile = await StorageFile.GetFileFromApplicationUriAsync(new Uri($"ms-appx:///Assets/mnist.onnx"));
            var model = await mnistModel.CreateFromStreamAsync(modelFile);

            // 组织输入
            var inputArray = await GetInputDataFromInk();
            var inputTensor = TensorFloat.CreateFromArray(new List<long> { 1, 784 }, inputArray);
            var modelInput = new mnistInput { fc1x = inputTensor };

            // 推理
            var result = await model.EvaluateAsync(modelInput);

            // 得到每个数字的得分
            var scoreList = result.activation3y.GetAsVectorView().ToList();

            // 从输出中取出得分最高的
            var max = scoreList.IndexOf(scoreList.Max());

            // 显示在控件中
            lbResult.Text = max.ToString();
        }
    }
}
