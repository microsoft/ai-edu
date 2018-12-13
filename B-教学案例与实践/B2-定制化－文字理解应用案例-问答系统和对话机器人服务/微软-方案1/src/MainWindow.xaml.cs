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

namespace QAClient
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();
        }

        private async void btn_Send_Click(object sender, RoutedEventArgs e)
        {
            // send http post request to qa service
            Answers results = await QAServiceAgent.DoQuery(this.tb_Question.Text);
            if (results.answers != null && results.answers.Length > 0)
            {
                this.tb_Dialog.Text += "问：" + this.tb_Question.Text + "\r\n";
                this.tb_Dialog.Text += results.answers[0].ToString() + "\r\n";
            }
        }
    }
}