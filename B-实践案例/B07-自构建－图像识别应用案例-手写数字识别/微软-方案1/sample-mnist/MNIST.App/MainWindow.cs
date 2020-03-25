// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using Model;// reference MNISTModelLibrary

namespace MNIST.App
{
    public partial class MainWindow : Form
    {
        public MainWindow()
        {
            InitializeComponent();
        }

        // 此为 Mnist 数据集的图片大小。推理时必须将图片变为同样大小。
        private const int ImageSize = 28;

        // 在窗体中声明模型类，以便在多次推理时能重用。
        private Mnist model;
        // 绘图对象，用于清除输入等。
        private Graphics graphics;
        // 每次画线的起始位置。
        private Point startPoint;

        /// <summary>
        /// 窗体的加载事件，在窗体显示时只执行一次。
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void Form1_Load(object sender, EventArgs e)
        {
            // 新建模型
            model = new Mnist();
            // 初始化手写区为其大小的位图，以便进行操作。
            writeArea.Image = new Bitmap(writeArea.Width, writeArea.Height);
            // 获取手写区位图的绘图类，以便以后重置图像。
            graphics = Graphics.FromImage(writeArea.Image);

            // 清除图像及文字。
            clear();
        }

        /// <summary>
        /// 此事件函数会在点击清除按钮时执行。
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void clean_click(object sender, EventArgs e)
        {
            // 清除图像及文字。
            clear();
        }

        /// <summary>
        /// 在手写区有鼠标按钮按下，或触摸屏下手指接触到屏幕时执行
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void writeArea_MouseDown(object sender, MouseEventArgs e)
        {
            // 鼠标事件较多，通过条件来仅在鼠标左键按下，或手指在屏幕上时，才执行。
            if (e.Button == MouseButtons.Left)
            {
                // 将鼠标当前位置保存起来，以便进行移动中的第一次画线。
                startPoint = e.Location;
            }
        }

        /// <summary>
        /// 手写区有鼠标移动，或触摸屏下手指在屏幕上移动时执行
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void writeArea_MouseMove(object sender, MouseEventArgs e)
        {
            // 鼠标事件较多，通过条件来仅在鼠标左键按下，或手指在屏幕上时，才执行。
            if (e.Button == MouseButtons.Left)
            {
                // 初始化画笔风格，包括：黑色，宽度40，起始及结束点均为圆头。
                // 这里也和 MNIST 训练时的数据格式有关。黑色是因为训练数据是黑白的，宽度40是为了笔画不要太细。
                // 起始及结束点是圆头，为了保证连续画出来的直线能够看起来更像曲线。如果不设置这个，会画出一些矩形，看起来不像是连续的笔画。
                Pen penStyle = new Pen(Color.Black, 40) { StartCap = LineCap.Round, EndCap = LineCap.Round };
                // 用上面的画笔来画一条起始位置到当前位置的直线。由于本移动事件会频繁的触发。因此，多条很短的直线看起来像曲线。
                graphics.DrawLine(penStyle, startPoint, e.Location);
                // 让手写区失效，从而触发重绘，更新当前区域。
                writeArea.Invalidate();
                // 在画完之后，将起始位置设置成当前位置，准备好作为下一笔的。
                startPoint = e.Location;
            }
        }

        /// <summary>
        /// 手写区松开鼠标键，或手指离开屏幕时执行
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void writeArea_MouseUp(object sender, MouseEventArgs e)
        {
            // 鼠标事件较多，通过条件来仅在鼠标左键按下，或手指在屏幕上时，才执行。
            if (e.Button == MouseButtons.Left)
            {
                /*
                 * 
                 * 这是数据规范化中较核心的代码，将手写图片转换为模型训练时的数据格式。
                 * 
                */

                // 1. 将手写区图片缩小至 28 x 28，与训练数据格式一致。
                Bitmap clonedBmp = new Bitmap(writeArea.Image, ImageSize, ImageSize);

                // 声明返回的二维数组。
                List<float> image = new List<float>(ImageSize * ImageSize);
                // 按行、列循环所有像素点，取得像素点做下一步操作。
                for (int y = 0; y < ImageSize; y++)
                {
                    for (int x = 0; x < ImageSize; x++)
                    {
                        Color color = clonedBmp.GetPixel(x, y);
                        // 规范化值
                        //    将数值的RGB通道取平均值
                        double average = (color.R + color.G + color.B) / 3.0;
                        //    将平均值范围缩小到0到1之间
                        double oneValue = average / 255;
                        //    将黑白翻转，并加上0.5，使大部分值为非零，并与训练时的数据格式一样
                        double reversed = 0.5 - oneValue;
                        image.Add((float)reversed);
                    }
                }

                // 2. 推理
                //    将数组放到另一层列表中，使其成为和推理函数一样的双层列表。
                long inferResult = model.Infer(new List<IEnumerable<float>> { image }).First().First();
                //    显示推理结果。
                outputText.Text = inferResult.ToString();
            }
        }

        /// <summary>
        /// 将手写区域清除为白色，并删除推理结果。
        /// </summary>
        private void clear()
        {
            // 用绘图类将手写区设置为白色。因为Mnist数据集实际上是黑白的，所以设置成白色，才能达到最好的识别效果。
            // 如果设置成其它显色，而其它地方的代码不做变化，将会降低识别率。
            graphics.Clear(Color.White);
            // 将手写区设置为失效，触发重绘来设置为白色。
            writeArea.Invalidate();
            // 清除输出的推理结果。
            outputText.Text = string.Empty;
        }
    }
}