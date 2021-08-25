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
using ExtendedModel; // 引用扩展后的模型

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

        // 当笔画在水平面上的投影相互重叠时，通过这个阈值来尽可能地实现数字的分割。
        private const double ProjectionOverlayRatioThreshold = 0.1;

        // 在窗体中声明模型类，以便在多次推理时能重用。
        private MnistExtension model;
        // 绘图对象，用于清除输入等。
        private Graphics graphics;
        // 每次画线的起始位置。
        private Point startPoint;

        // 当前
        private List<Point> strokePoints = new List<Point>();
        private List<StrokeRecord> allStrokes = new List<StrokeRecord>();

        /// <summary>
        /// 窗体的加载事件，在窗体显示时只执行一次。
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void Form1_Load(object sender, EventArgs e)
        {
            // 新建模型
            model = new MnistExtension();
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

            // 清除记录了的笔画。
            allStrokes.Clear();
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

                // 每次鼠标点下时，开始重新记录鼠标移动点。
                strokePoints.Clear();
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
                // 初始化画笔风格，包括：黑色，宽度20，起始及结束点均为圆头。
                // 这里也和 MNIST 训练时的数据格式有关。黑色是因为训练数据是黑白的，宽度40是为了笔画不要太细。
                // 起始及结束点是圆头，为了保证连续画出来的直线能够看起来更像曲线。如果不设置这个，会画出一些矩形，看起来不像是连续的笔画。
                Pen penStyle = new Pen(Color.Black, 20) { StartCap = LineCap.Round, EndCap = LineCap.Round };
                // 用上面的画笔来画一条起始位置到当前位置的直线。由于本移动事件会频繁的触发。因此，多条很短的直线看起来像曲线。
                graphics.DrawLine(penStyle, startPoint, e.Location);
                // 让手写区失效，从而触发重绘，更新当前区域。
                writeArea.Invalidate();
                // 在画完之后，将起始位置设置成当前位置，准备好作为下一笔的。
                startPoint = e.Location;

                // 记录鼠标移动点。
                strokePoints.Add(e.Location);
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
                // 必须确实发生了鼠标移动事件，即有线段被画出，我们才认为有笔画存在。
                if (strokePoints.Any())
                {
                    var thisStrokeRecord = new StrokeRecord(strokePoints);
                    allStrokes.Add(thisStrokeRecord);

                    // 将所有笔画按水平起点排序，然后按重叠与否进行分组。
                    allStrokes = allStrokes.OrderBy(s => s.HorizontalStart).ToList();
                    int[] strokeGroupIds = new int[allStrokes.Count];
                    int nextGroupId = 1;

                    for (int i = 0; i < allStrokes.Count; i++)
                    {
                        // 为了避免水平方向太多笔画被连在一起，我们采取一种简单的办法：
                        // 当1、2笔画重叠时，我们就不会在检查笔画2和更右侧笔画是否重叠。
                        if (strokeGroupIds[i] != 0)
                        {
                            continue;
                        }

                        strokeGroupIds[i] = nextGroupId;
                        nextGroupId++;

                        var s1 = allStrokes[i];
                        for (int j = 1; i + j < allStrokes.Count; j++)
                        {
                            var s2 = allStrokes[i + j];

                            if (s2.HorizontalStart < s1.OverlayMaxStart)
                            {
                                if (strokeGroupIds[i + j] == 0)
                                {
                                    if (s1.OverlayWith(s2))
                                    {
                                        strokeGroupIds[i + j] = strokeGroupIds[i];
                                    }
                                }
                            }
                            else
                            {
                                break;
                            }
                        }
                    }

                    bool enableDebug = visualizeSwitch.Checked;
                    if (enableDebug)
                    {
                        graphics.Clear(Color.White);
                    }

                    // 清除之前显式的推理结果
                    outputText.Text = "";

                    var batchInferInput = new List<IEnumerable<float>>();

                    Pen penStyle = new Pen(Color.Black, 20)
                    {
                        StartCap = LineCap.Round,
                        EndCap = LineCap.Round
                    };

                    List<IGrouping<int, StrokeRecord>> groups = allStrokes
                        .Zip(strokeGroupIds, Tuple.Create)
                        .GroupBy(tuple => tuple.Item2, tuple => tuple.Item1) // Item2是分组编号, Item1是StrokeRecord
                        .ToList();
                    foreach (IGrouping<int, StrokeRecord> group in groups)
                    {
                        int gid = group.Key;
                        var groupedStrokes = group.ToList(); // IGrouping<TKey, TElement>本质上也是一个可迭代的IEnumerable<TElement>

                        // 确定整个分组的所有笔画的范围。
                        int grpHorizontalStart = groupedStrokes.Min(s => s.HorizontalStart);
                        int grpHorizontalEnd = groupedStrokes.Max(s => s.HorizontalEnd);
                        int grpHorizontalLength = grpHorizontalEnd - grpHorizontalStart;

                        int canvasEdgeLen = writeArea.Height;
                        Bitmap canvas = new Bitmap(canvasEdgeLen, canvasEdgeLen);
                        Graphics canvasGraphics = Graphics.FromImage(canvas);
                        canvasGraphics.Clear(Color.White);

                        // 因为我们提取了每个笔画，就不能把长方形的绘图区直接当做输入了。
                        // 这里我们把宽度小于 writeArea.Height 的分组在 canvas 内居中。
                        int halfOffsetX = Math.Max(canvasEdgeLen - grpHorizontalLength, 0) / 2;

                        var grpClr = GetDebugColor(gid);
                        var rectClr = Color.FromArgb(120, grpClr);

                        foreach (var stroke in groupedStrokes)
                        {
                            if (enableDebug)
                            {
                                graphics.FillRectangle(
                                    new SolidBrush(rectClr),
                                    stroke.OverlayMinEnd,
                                    0,
                                    Math.Max(2, stroke.OverlayMaxStart - stroke.OverlayMinEnd), // At least width of 2px
                                    30);
                            }

                            Point startPoint = stroke.Points[0];
                            foreach (var point in stroke.Points.Skip(1))
                            {
                                var from = startPoint;
                                var to = point;

                                // 因为每个分组都是在长方形的绘图区被记录的，所以在单一位图上，需要先减去相对于长方形绘图区的偏移量 grpHorizontalStart
                                from.X = from.X - grpHorizontalStart + halfOffsetX;
                                to.X = to.X - grpHorizontalStart + halfOffsetX;
                                canvasGraphics.DrawLine(penStyle, from, to);

                                /*
                                 * 调试用。
                                 * 取消注释后可以看到每一笔画，会按照其分组显示不同的颜色。
                                 */
                                if (enableDebug)
                                {
                                    graphics.DrawLine(
                                        new Pen(grpClr, 20)
                                        {
                                            StartCap = LineCap.Round,
                                            EndCap = LineCap.Round
                                        },
                                        startPoint,
                                        point);
                                }

                                startPoint = point;
                            }
                        }

                        // 1. 将分割出的笔画图片缩小至 28 x 28，与训练数据格式一致。
                        Bitmap clonedBmp = new Bitmap(canvas, ImageSize, ImageSize);

                        var image = new List<float>(ImageSize * ImageSize);
                        for (var x = 0; x < ImageSize; x++)
                        {
                            for (var y = 0; y < ImageSize; y++)
                            {
                                var color = clonedBmp.GetPixel(y, x);
                                image.Add((float)(0.5 - (color.R + color.G + color.B) / (3.0 * 255)));
                            }
                        }

                        // 将这一组笔画对应的矩阵保存下来，以备批量推理。
                        batchInferInput.Add(image);
                    }

                    // 2. 进行批量推理
                    //    batchInferInput 是一个列表，它的每个元素都是一次推量的输入。
                    //IEnumerable<IEnumerable<long>> inferResult = model.Infer(batchInferInput);
                    var inferResult = batchInferInput.SelectMany(i => model.Infer(new List<IEnumerable<float>> { i })).ToList();

                    //    推量的结果是一个可枚举对象，它的每个元素代表了批量推理中一次推理的结果。我们用 仅一次.First() 将它们的结果都取出来，并格式化。
                    // outputText.Text = string.Join("", inferResult.Select(singleResult => singleResult.First().ToString()));

                    var recognizedLabels = inferResult.Select(singleResult => (int)singleResult.First()).ToList();
                    outputText.Text = EvaluateAndFormatExpression(recognizedLabels);

                    if (enableDebug)
                    {
                        // 这是调试用的。在上面的调试代码没有启用时，这句话没有特别作用。
                        writeArea.Invalidate();
                    }
                }
            }
        }

        private string EvaluateAndFormatExpression(List<int> recognizedLabels)
        {
            string[] operatorsToEval = { "+", "-", "*", "/", "(", ")" };
            string[] operatorsToDisplay = { "+", "-", "×", "÷", "(", ")" };

            string toEval = string.Join("", recognizedLabels.Select(label =>
            {
                if (0 <= label && label <= 9)
                {
                    return label.ToString();
                }

                return operatorsToEval[label - 10];
            }));

            string result = "Error";
            try
            {
                var evalResult = new DataTable().Compute(toEval, null);
                if (evalResult is double)
                {
                    result = ((double)evalResult).ToString("F4");
                }
                else if (!(evalResult is DBNull))
                {
                    result = evalResult.ToString();
                }
            }
            catch (SyntaxErrorException)
            {
            }

            string toDisplay = string.Join("", recognizedLabels.Select(label =>
            {
                if (0 <= label && label <= 9)
                {
                    return label.ToString();
                }

                return operatorsToDisplay[label - 10];
            }));

            return $"{toDisplay}={result}";

        }

        private Color GetDebugColor(int gid)
        {
            // gid 从1开始
            Color[] strokeColorsForDebugging = { Color.Red, Color.Blue, Color.YellowGreen, Color.Gray, Color.Green, Color.Black, Color.Pink, Color.Purple, Color.Orange, Color.Aqua };
            return strokeColorsForDebugging[(gid - 1) % strokeColorsForDebugging.Length];
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

        /// <summary>
        /// 用于记录历史笔画信息的数据结构。
        /// </summary>
        class StrokeRecord
        {
            public StrokeRecord(List<Point> strokePoints)
            {
                // 拷贝所有Point以避免列表在外部被修改。
                Points = new List<Point>(strokePoints);

                HorizontalStart = Points.Min(pt => pt.X);
                HorizontalEnd = Points.Max(pt => pt.X);
                HorizontalLength = HorizontalEnd - HorizontalStart;

                OverlayMaxStart = HorizontalStart + (int)(HorizontalLength * (1 - ProjectionOverlayRatioThreshold));
                OverlayMinEnd = HorizontalStart + (int)(HorizontalLength * ProjectionOverlayRatioThreshold);
            }

            /// <summary>
            /// 构成这一笔画的点。
            /// </summary>
            public List<Point> Points { get; }

            /// <summary>
            /// 这一笔画在水平方向上的起点。
            /// </summary>
            public int HorizontalStart { get; }

            /// <summary>
            /// 这一笔画在水平方向上的终点。
            /// </summary>
            public int HorizontalEnd { get; }

            /// <summary>
            /// 这一笔画在水平方向上的长度。
            /// </summary>
            public int HorizontalLength { get; }

            /// <summary>
            /// 另一笔画必须越过这些阈值点，才被认为和这一笔画重合。
            /// </summary>
            public int OverlayMaxStart { get; }
            public int OverlayMinEnd { get; }

            /// <summary>
            /// 检查另一笔画是否“单方面”被认为和这一笔画重叠。这个检查不是对称关系。
            /// </summary>
            /// <param name="other"></param>
            private bool CheckPosition(StrokeRecord other)
            {
                return (other.HorizontalStart < OverlayMaxStart) || (OverlayMinEnd < other.HorizontalEnd);
            }

            public bool OverlayWith(StrokeRecord other)
            {
                return this.CheckPosition(other) || other.CheckPosition(this);
            }
        }
    }
}