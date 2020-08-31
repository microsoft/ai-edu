using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Windows.Forms;

using System.Net;
using System.Net.Http;
using Newtonsoft.Json;
using System.Threading.Tasks;
using System.Threading;

namespace VideoLabels
{
    public partial class VideoLabel : Form
    {
        private string apiUrl = "https://api.videoindexer.ai";
        private string location = "trial";
        private string apiKey = string.Empty;
        private string accountID = string.Empty;
        private string accountAccessToken = string.Empty;

        private HttpClient client;

        public VideoLabel()
        {
            InitializeComponent();
        }

        #region Initialize
        private void Initialize()
        {
            InitializeClient();
            InitializeVIAccess();
        }
        // 初始化网络连接
        private void InitializeClient()
        {
            ServicePointManager.SecurityProtocol =
                    ServicePointManager.SecurityProtocol | SecurityProtocolType.Tls12;

            // create the http client
            var handler = new HttpClientHandler();
            handler.AllowAutoRedirect = false;
            client = new HttpClient(handler);

            ShowLog("初始化连接完成！");
        }

        // 初始化 Video Indexer 连接，获取 Account Access Token
        private void InitializeVIAccess()
        {
            GetAccountAccessToken(out accountAccessToken);
        }
        #endregion

        #region Event
        // 设定 Account ID 和 Primary Key
        private void Button_Setting_Click(object sender, EventArgs e)
        {
            string strID = textBox_AccountID.Text;
            string strKEY = textBox_PrimaryKey.Text;

            if (string.IsNullOrEmpty(strID) 
                || string.IsNullOrEmpty(strKEY))
            {
                MessageBox.Show("请输入有效 ID 和 KEy！");
                return;
            }

            accountID = strID;
            apiKey = strKEY;

            // 异步初始化，避免阻塞UI线程
            Task.Run(() => Initialize());
        }

        private void Button_Select_Click(object sender, EventArgs e)
        {
            string filePath;
            GetSelectFilePath(out filePath);
            textBox_FilePath.Text = filePath;

            ShowLog("选择文件：" + filePath);
        }

        private void Button_Upload_Click(object sender, EventArgs e)
        {
            string filePath = textBox_FilePath.Text;
            if (string.IsNullOrEmpty(filePath))
            {
                MessageBox.Show("请先选择视频文件！");
                return;
            }

            if (string.IsNullOrEmpty(accountAccessToken))
            {
                MessageBox.Show("请等待初始化完成！");
                return;
            }

            ShowLog("视频开始上传和处理，请不要重复点击上传按钮！");

            // 异步操作，避免阻塞UI线程
            Task.Run(() => UploadAndGetIndex(filePath));
        }

        private void Button_GetVideosID_Click(object sender, EventArgs e)
        {
            // 异步操作，避免阻塞UI线程
            Task.Run(() => ListVideos());
        }

        private void Button_GetLabelsWithID_Click(object sender, EventArgs e)
        {
            var videoID = textBox_VideoID.Text;
            if(string.IsNullOrEmpty(videoID))
            {
                MessageBox.Show("请输入正确的视频ID！");
                return;
            }

            // 异步操作，避免阻塞UI线程
            Task.Run(() => GetLabelsWithID(videoID));
        }
        #endregion

        #region Functional
        private void GetAccountAccessToken(out string accountAccessToken)
        {
            ShowLog("开始获取 Account Access Token！");

            client.DefaultRequestHeaders.Add("Ocp-Apim-Subscription-Key", apiKey);

            accountAccessToken = string.Empty;
            try
            {
                // obtain account access token
                var accountAccessTokenRequestResult = client.GetAsync(
                    $"{apiUrl}/auth/{location}/Accounts/{accountID}/AccessToken?allowEdit=true"
                    ).Result;
                accountAccessToken = accountAccessTokenRequestResult.Content.ReadAsStringAsync().Result.Replace("\"", "");

            }
            catch (Exception e)
            {
                MessageBox.Show("获取 Account Access Token 错误！请确认 ID 和 Key 后重新初始化！" + e.Message);
            }
            
            client.DefaultRequestHeaders.Remove("Ocp-Apim-Subscription-Key");

            ShowLog("获取 Account Access Token 完毕！");
        }

        // 对某个视频的操作，需要先获取到 Video Access Token
        private void GetVideoAccessToken(string videoID, out string videoAccessToken)
        {
            ShowLog("开始获取 Video Access Token！");

            // obtain video access token            
            client.DefaultRequestHeaders.Add("Ocp-Apim-Subscription-Key", apiKey);

            videoAccessToken = string.Empty;

            try
            {
                var videoTokenRequestResult = client.GetAsync(
                $"{apiUrl}/auth/{location}/Accounts/{accountID}/Videos/{videoID}/AccessToken?allowEdit=true"
                ).Result;
                videoAccessToken = videoTokenRequestResult.Content.ReadAsStringAsync().Result.Replace("\"", "");
            }
            catch (Exception e)
            {
                MessageBox.Show("获取 Video Access Token 错误！请确认 Video ID 后重新操作！" + e.Message);
            }
            

            client.DefaultRequestHeaders.Remove("Ocp-Apim-Subscription-Key");

            ShowLog("获取 Video Access Token 完毕！");
        }

        // 获取账户下的所有视频
        private void ListVideos()
        {
            ShowLog("开始获取视频ID列表！");

            try
            {
                var videoListRequestResult = client.GetAsync(
                $"{apiUrl}/{location}/Accounts/{accountID}/Videos?accessToken={accountAccessToken}"
                ).Result;
                var videoListJson = videoListRequestResult.Content.ReadAsStringAsync().Result;

                // get the video id from the upload result
                var videoContentList = JsonConvert.DeserializeObject<dynamic>(videoListJson)["results"];

                StringBuilder strBulder = new StringBuilder();
                for (int i = 0; i < videoContentList.Count - 1; i++)
                {
                    strBulder.AppendLine(videoContentList[i]["id"].Value);
                }
                strBulder.Append(videoContentList[videoContentList.Count - 1]["id"].Value);

                var videoList = strBulder.ToString();
                ShowVideoIDList(videoList, false);
            }
            catch (Exception e)
            {
                MessageBox.Show("获取 Video 列表错误！" + e.Message);
                return;
            }
            

            ShowLog("获取视频ID列表完成！");
        }

        // 获取视频见解
        private string GetVideoIndex(string videoID, string videoAccessToken)
        {
            ShowLog("开始获取 Video Index！");

            string videoGetIndexResult = string.Empty;

            try
            {
                var videoGetIndexRequestResult = client.GetAsync(
                    $"{apiUrl}/{location}/Accounts/{accountID}/Videos/{videoID}/Index?accessToken={videoAccessToken}&language=English"
                    ).Result;
                videoGetIndexResult = videoGetIndexRequestResult.Content.ReadAsStringAsync().Result;

            }
            catch (Exception e)
            {
                MessageBox.Show("获取视频见解错误" + e.Message);
                return string.Empty;
            }
            
            ShowLog("获取 Video Index 完毕！");

            return videoGetIndexResult;
        }

        // 根据视频ID 获取视频标签
        private void GetLabelsWithID(string videoID)
        {
            string videoAccessToken = string.Empty;
            string videoGetIndexResult = string.Empty;

            GetVideoAccessToken(videoID, out videoAccessToken);

            if (!string.IsNullOrEmpty(videoAccessToken))
            {
                videoGetIndexResult = GetVideoIndex(videoID, videoAccessToken);

                if (!string.IsNullOrEmpty(videoGetIndexResult))
                {
                    OutputLabels(videoGetIndexResult);
                }
            }
        }

        // 输出视频标签
        private void OutputLabels(string strJson)
        {
            StringBuilder strBulder = new StringBuilder();

            try
            {
                var labelsFromJson = JsonConvert.DeserializeObject<dynamic>(strJson)["summarizedInsights"]["labels"];
                
                for (int i = 0; i < labelsFromJson.Count; i++)
                {
                    strBulder.AppendLine(labelsFromJson[i]["name"].Value);
                }
            }
            catch (Exception e)
            {
                MessageBox.Show("解析视频标签错误！" + e.Message);
            }

            ShowLabels(strBulder.ToString());
        }

        // 上传视频，并进行索引，得到视频见解
        private void UploadAndGetIndex(string filePath)
        {
            var videoID = UploadContent(filePath);
            if (string.IsNullOrEmpty(videoID))
            {
                MessageBox.Show("视频上传错误，请重新上传！");
                return;
            }

            ShowVideoIDList(videoID, true);

            string videoAccessToken = string.Empty;
            GetVideoAccessToken(videoID, out videoAccessToken);

            ShowLog("开始处理视频！");

            // wait for the video index to finish
            while (true)
            {
                Thread.Sleep(10000);

                var videoGetIndexResult = GetVideoIndex(videoID, videoAccessToken);

                try
                {
                    var processingState = JsonConvert.DeserializeObject<dynamic>(videoGetIndexResult)["state"];

                    ShowLog("视频处理状态:" + processingState.Value);

                    // job is finished
                    if (processingState != "Uploaded" && processingState != "Processing")
                    {
                        ShowLog("视频处理完毕！");
                        OutputLabels(videoGetIndexResult);
                        break;
                    }
                }
                catch (Exception e)
                {
                    MessageBox.Show("解析视频处理状态错误！" + e.Message);
                    return;
                }
            }
        }

        // 文件选择
        private void GetSelectFilePath(out string filePath)
        {
            filePath = string.Empty;
            using (OpenFileDialog openFileDialog = new OpenFileDialog())
            {
                openFileDialog.InitialDirectory = "C:\\";
                openFileDialog.Filter = "Video files (*.mp4;*.mov)|*.mp4;*.mov";
                openFileDialog.FilterIndex = 1;
                openFileDialog.RestoreDirectory = true;

                if (openFileDialog.ShowDialog() == DialogResult.OK)
                {
                    filePath = openFileDialog.FileName;
                }
            }
        }

        // 上传视频
        private string UploadContent(string filePath)
        {
            MultipartFormDataContent content = new MultipartFormDataContent();

            try
            {
                FileStream video = File.OpenRead(filePath);
                byte[] buffer = new byte[video.Length];
                video.Read(buffer, 0, buffer.Length);
                content.Add(new ByteArrayContent(buffer));
            }
            catch (Exception e)
            {
                MessageBox.Show("文件读取错误！" + e.Message);
                return string.Empty;
            }
            

            ShowLog("读取文件完成，开始上传！");

            string videoID = string.Empty;
            try
            {
                var uploadRequestResult = client.PostAsync(
                $"{apiUrl}/{location}/Accounts/{accountID}" +
                $"/Videos?accessToken={accountAccessToken}" +
                $"&name=some_name" +
                $"&description=some_description" +
                $"&privacy=private" +
                $"&partition=some_partition",
                content
                ).Result;
                var uploadResult = uploadRequestResult.Content.ReadAsStringAsync().Result;

                // get the video id from the upload result
                videoID = JsonConvert.DeserializeObject<dynamic>(uploadResult)["id"];
            }
            catch (Exception e)
            {
                MessageBox.Show("文件上传错误！" + e.Message);
                return string.Empty;
            }
            

            ShowLog("上传完成！");
            ShowLog("Video ID: " + videoID);

            return videoID;
        }

        // 显示状态
        private void ShowLog(string strMsg, bool IsAppend = true)
        {
            strMsg = DateTime.Now.ToString("hh:mm:ss") + ":" + strMsg;
            ShowTextBoxText(strMsg, IsAppend, textBox_Status);
        }

        // 显示视频标签列表
        private void ShowLabels(string Labels, bool IsAppend = false)
        {
            ShowTextBoxText(Labels, IsAppend, textBox_VideoLabels);
        }

        // 显示视频ID 列表
        private void ShowVideoIDList(string videoIDList, bool IsAppend)
        {
            ShowTextBoxText(videoIDList, IsAppend, textBox_VideoIDList);
        }

        private void ShowTextBoxText(string videoIDList, bool IsAppend, TextBox tb)
        {
            // 由于非UI线程不能操作UI元素，因此采用以下方式调用
            BeginInvoke(new MethodInvoker(() =>
            {
                if (IsAppend)
                {
                    tb.AppendText(videoIDList + Environment.NewLine);
                }
                else
                {
                    tb.Text = videoIDList + Environment.NewLine;
                }
            }));
        }
        #endregion

    }
}
