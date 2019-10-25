using Microsoft.Win32.SafeHandles;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Net.Http;
using System.IO;
using System.Media;
using System.Windows.Forms.VisualStyles;

namespace TTS_Demo
{
    

    public partial class Form1 : Form
    {
        
        string tempFile = "temp.wav"; //临时文件存储路径

        TTSApi tts = new TTSApi(); 

        public Form1()
        {
            InitializeComponent();

        }

        //点击“转换”按钮
        private async void transferButton_Click(object sender, EventArgs e)
        {

            string text = textBox1.Text;
            
            if (text.Length > 0)
            {
                tips.Text = "语音生成中...";

                await tts.textToSpeechAsync(text, tempFile);
                
                tips.Text = "";
                playButton.Enabled = true;
                saveButton.Enabled = true;
            }

        }

        //点击“播放”按钮
        private void playButton_Click(object sender, EventArgs e)
        {
            SoundPlayer playSound = new SoundPlayer(tempFile);
            playSound.Play();
        }

        //点击“保存”按钮
        private void saveButton_Click(object sender, EventArgs e)
        {
            string filePath = "";
            //取前10个字符作为文件名
            string fileName = (textBox1.Text.Length < 10) ? textBox1.Text : textBox1.Text.Substring(0, 10);

            SaveFileDialog saveFile = new SaveFileDialog();
            saveFile.FileName = fileName;
            saveFile.Filter = "音频文件 (*.wav) | *.wav";
            //保存对话框是否记忆上次打开的目录 
            saveFile.RestoreDirectory = true;

            //点了保存按钮进入 
            if (saveFile.ShowDialog() == DialogResult.OK)
            {
                filePath = saveFile.FileName.ToString(); //获得文件路径

                if (File.Exists(tempFile))
                {
                    File.Copy(tempFile, filePath, true);
                }
                else
                {
                    Console.WriteLine("音频文件不存在");
                }
            }
        }

        private void textBox1_TextChanged(object sender, EventArgs e)
        {
            playButton.Enabled = false;
            saveButton.Enabled = false;
        }
    }
}
