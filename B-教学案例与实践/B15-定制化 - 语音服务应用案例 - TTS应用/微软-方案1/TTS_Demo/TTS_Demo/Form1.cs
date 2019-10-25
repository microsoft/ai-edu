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

        string accessToken;
        Authentication auth = new Authentication("https://westus.api.cognitive.microsoft.com/sts/v1.0/issuetoken","5f7e7c7254ef4415bd14950350ebc0c5");
        string host = "https://westus.tts.speech.microsoft.com/cognitiveservices/v1";

        public Form1()
        {
            InitializeComponent();

        }

        //点击“转换”按钮
        private async void transferButton_Click(object sender, EventArgs e)
        {

            playButton.Enabled = false;
            saveButton.Enabled = false ;
            tips.Text = "语音生成中...";

            string text = textBox1.Text;

            await textToSpeechAsync(text,tempFile);

            tips.Text = "";
            playButton.Enabled = true;
            saveButton.Enabled = true;

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
            saveSpeech();
        }


        //转换文本并保存
        private async Task textToSpeechAsync(string text,string savePath)
        {
            try
            {
                accessToken = await auth.FetchTokenAsync().ConfigureAwait(false);
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex);
            }

            string body = @"<speak version='1.0' xmlns='https://www.w3.org/2001/10/synthesis' xml:lang='en-US'>
              <voice name='Microsoft Server Speech Text to Speech Voice (en-US, ZiraRUS)'>" +
              text + "</voice></speak>";

            using (var client = new HttpClient())
            {
                using (var request = new HttpRequestMessage())
                {
                    // Set the HTTP method
                    request.Method = HttpMethod.Post;
                    // Construct the URI
                    request.RequestUri = new Uri(host);
                    // Set the content type header
                    request.Content = new StringContent(body, Encoding.UTF8, "application/ssml+xml");
                    // Set additional header, such as Authorization and User-Agent
                    request.Headers.Add("Authorization", "Bearer " + accessToken);
                    request.Headers.Add("Connection", "Keep-Alive");
                    // Update your resource name
                    request.Headers.Add("User-Agent", "YOUR_RESOURCE_NAME");
                    request.Headers.Add("X-Microsoft-OutputFormat", "riff-24khz-16bit-mono-pcm");
                    // Create a request
                    Console.WriteLine("Calling the TTS service. Please wait... \n");
                    using (var response = await client.SendAsync(request).ConfigureAwait(false))
                    {
                        response.EnsureSuccessStatusCode();
                        // Asynchronously read the response
                        using (var dataStream = await response.Content.ReadAsStreamAsync().ConfigureAwait(false))
                        {
                            Console.WriteLine("savePath:"+savePath);
                            using (var fileStream = new FileStream(savePath, FileMode.Create, FileAccess.Write, FileShare.Write))
                            {
                                await dataStream.CopyToAsync(fileStream).ConfigureAwait(false);
                                fileStream.Close();
                            }
     
                        }
                    }
                }
            }

        }

        //存储生成的音频
        private void saveSpeech()
        {
            string filePath = "";

            string fileName = textBox1.Text.Substring(0,10);

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
                    File.Copy(tempFile, filePath , true);
                }
            }
  
        }


    }
}
