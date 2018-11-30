// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace CartoonTranslate
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        private string Version;
        private string Language;
        private OcrResult.Rootobject ocrResult;
        private NewOcrResult.Rootobject newOcrResult;

        public MainWindow()
        {
            InitializeComponent();
        }

        private void btn_Show_Click(object sender, RoutedEventArgs e)
        {
            if (!Uri.IsWellFormedUriString(this.tb_Url.Text, UriKind.Absolute))
            {
                // show warning message
                return;
            }

            // show image at imgSource
            BitmapImage bi = new BitmapImage();
            bi.BeginInit();
            bi.UriSource = new Uri(this.tb_Url.Text);
            bi.EndInit();
            this.imgSource.Source = bi;
            this.imgTarget.Source = bi;
        }

        private async void btn_OCR_Click(object sender, RoutedEventArgs e)
        {
            this.Version = GetVersion();
            this.Language = GetLanguage();

            if (Version == "OCR")
            {
                ocrResult = await CognitiveServiceAgent.DoOCR(this.tb_Url.Text, Language);
                foreach (OcrResult.Region region in ocrResult.regions)
                {
                    foreach (OcrResult.Line line in region.lines)
                    {
                        if (line.Convert())
                        {
                            Rectangle rect = new Rectangle()
                            {
                                Margin = new Thickness(line.BB[0], line.BB[1], 0, 0),
                                Width = line.BB[2],
                                Height = line.BB[3],
                                Stroke = Brushes.Red,
                                //Fill =Brushes.White
                            };
                            this.canvas_1.Children.Add(rect);
                        }
                    }
                }
            }
            else
            {
                newOcrResult = await CognitiveServiceAgent.DoRecognizeText(this.tb_Url.Text);
                // 1 - erase the original text
                foreach (NewOcrResult.Line line in newOcrResult.recognitionResult.lines)
                {
                    Polygon p = new Polygon();
                    PointCollection pc = new PointCollection();
                    pc.Add(new Point(line.boundingBox[0], line.boundingBox[1]));
                    pc.Add(new Point(line.boundingBox[2], line.boundingBox[3]));
                    pc.Add(new Point(line.boundingBox[4], line.boundingBox[5]));
                    pc.Add(new Point(line.boundingBox[6], line.boundingBox[7]));
                    p.Points = pc;
                    p.Stroke = Brushes.Red;
                    this.canvas_1.Children.Add(p);
                }
            }
        }

        private async void btn_Translate_Click(object sender, RoutedEventArgs e)
        {
            List<string> listTarget = await this.Translate();
            this.ShowText(listTarget);
        }
        
        private async Task<List<string>> Translate()
        {
            List<string> listSource = new List<string>();
            List<string> listTarget = new List<string>();
            if (this.Version == "OCR")
            {
                foreach (OcrResult.Region region in ocrResult.regions)
                {
                    foreach (OcrResult.Line line in region.lines)
                    {
                        listSource.Add(line.TEXT);
                        if (listSource.Count == 25)
                        {
                            List<string> listOutput = await CognitiveServiceAgent.DoTranslate(listSource, Language, "zh-Hans");
                            listTarget.AddRange(listOutput);
                            listSource.Clear();
                        }
                    }
                }
                if (listSource.Count > 0)
                {
                    List<string> listOutput = await CognitiveServiceAgent.DoTranslate(listSource, Language, "zh-Hans");
                    listTarget.AddRange(listOutput);
                }
            }

            return listTarget;
        }

        private void ShowText(List<string> listTarget)
        {
            int i = 0;
            foreach (OcrResult.Region region in ocrResult.regions)
            {
                foreach (OcrResult.Line line in region.lines)
                {
                    string translatedLine = listTarget[i];

                    Rectangle rect = new Rectangle()
                    {
                        Margin = new Thickness(line.BB[0], line.BB[1], 0, 0),
                        Width = line.BB[2],
                        Height = line.BB[3],
                        Stroke = null,
                        Fill =Brushes.White
                    };
                    this.canvas_2.Children.Add(rect);

                    TextBlock tb = new TextBlock()
                    {
                        Margin = new Thickness(line.BB[0], line.BB[1], 0, 0),
                        Height = line.BB[3],
                        Width = line.BB[2],
                        Text = translatedLine,
                        FontSize = 16,
                        TextWrapping = TextWrapping.Wrap,
                        Foreground = Brushes.Red
                    };
                    this.canvas_2.Children.Add(tb);
                    i++;
                }
            }
        }

        private string GetLanguage()
        {
            if (this.rb_English.IsChecked == true)
            {
                return "en";
            }
            else
            {
                return "ja";
            }
        }

        private string GetVersion()
        {
            if (this.rb_V1.IsChecked == true)
            {
                return "OCR";
            }
            else
            {
                return "RecText";
            }
        }

        private void rb_V1_Click(object sender, RoutedEventArgs e)
        {
            this.rb_Japanese.IsEnabled = true;
        }

        private void rb_V2_Click(object sender, RoutedEventArgs e)
        {
            this.rb_English.IsChecked = true;
            this.rb_Japanese.IsChecked = false;
            this.rb_Japanese.IsEnabled = false;
        }

        private void btn_Cluster_Click(object sender, RoutedEventArgs e)
        {
            if (this.Version == "V1")
            {
                List<Rect> listRect = new List<Rect>();

                foreach (OcrResult.Region region in ocrResult.regions)
                {
                    foreach (OcrResult.Line line in region.lines)
                    {
                        Rect rect = new Rect()
                        {
                            X = line.BB[0],
                            Y = line.BB[1],
                            Width = line.BB[2],
                            Height = line.BB[3]
                        };
                        listRect.Add(rect);
                    }
                }

                Cluster c = new Cluster();
                List<Rect> listCluster = c.Dok(listRect);

                foreach(Rect rectC in listCluster)
                {
                    Rectangle rect = new Rectangle()
                    {
                        Margin = new Thickness(rectC.X, rectC.Y, 0, 0),
                        Width = rectC.Width,
                        Height = rectC.Height,
                        Stroke = Brushes.Blue,
                        StrokeThickness = 4
                        //Fill =Brushes.White
                    };
                    this.canvas_1.Children.Add(rect);

                }
            }
        }
    }

}