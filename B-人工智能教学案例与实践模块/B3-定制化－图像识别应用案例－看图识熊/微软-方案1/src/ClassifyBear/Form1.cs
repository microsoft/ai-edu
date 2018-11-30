// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace ClassifyBear
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
        }

        // 使用Netron查看模型，得到模型的输入应为227*227大小的图片
        private const int imageSize = 227;

        // 模型推理类
        private Model.Bear model;

        private void Form1_Load(object sender, EventArgs e)
        {
            // 初始化模型推理对象
            model = new Model.Bear();
        }

        private void button1_Click(object sender, EventArgs e)
        {
            // 识别之前先重置界面显示的内容
            label1.Text = string.Empty;
            pictureBox1.Image = null;
            pictureBox1.Refresh();

            bool isSuccess = false;
            try
            {
                pictureBox1.Load(textBox1.Text);
                isSuccess = true;
            }
            catch (Exception ex)
            {
                MessageBox.Show($"读取图片时出现错误：{ex.Message}");
                throw;
            }

            if (isSuccess)
            {
                // 图片加载成功后，从图片控件中取出227*227的位图对象
                Bitmap bitmap = new Bitmap(pictureBox1.Image, imageSize, imageSize);

                float[] imageArray = new float[imageSize * imageSize * 3];

                // 按照先行后列的方式依次取出图片的每个像素值
                for (int y = 0; y < imageSize; y++)
                {
                    for (int x = 0; x < imageSize; x++)
                    {
                        var color = bitmap.GetPixel(x, y);

                        // 使用Netron查看模型的输入发现
                        // 需要依次放置227 *227的蓝色分量、227*227的绿色分量、227*227的红色分量
                        imageArray[y * imageSize + x] = color.B;
                        imageArray[y * imageSize + x + 1* imageSize * imageSize] = color.G;
                        imageArray[y * imageSize + x + 2* imageSize * imageSize] = color.R;
                    }
                }

                // 模型推理类库支持一次推理多张图片，这里只使用一张图片
                var inputImages = new List<float[]>();
                inputImages.Add(imageArray);

                // 推理结果的第一个First()是取第一张图片的结果
                // 之前定义的输出只有classLabel，所以第二个First()就是分类的名字
                label1.Text = model.Infer(inputImages).First().First();
            }
        }
    }
}