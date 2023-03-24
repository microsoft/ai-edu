// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System;
using System.Threading.Tasks;
using System.Windows.Forms;
using Microsoft.CognitiveServices.Speech;
using Microsoft.Azure.CognitiveServices.Language.LUIS.Runtime;

namespace LightControl
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
            pictureBox1.Load("LightOff.png");
        }

        // 设置语音服务密钥及区域
        const string speechKey = "********************************";
        const string speechRegion = "******";

        // 设置语言理解服务终结点、密钥、应用程序ID
        const string luisEndpoint = "https://******.api.cognitive.microsoft.com";
        const string luisKey = "********************************";
        const string luisAppId = "********-****-****-****-************";

        // 语音识别器
        SpeechRecognizer recognizer;

        // 意图预测器
        Prediction intentPrediction;

        private void Form1_Load(object sender, EventArgs e)
        {
            try
            {
                SpeechConfig config = SpeechConfig.FromSubscription(speechKey, speechRegion);
                config.SpeechRecognitionLanguage = "zh-cn";
                recognizer = new SpeechRecognizer(config);

                // 挂载识别中的事件
                // 收到中间结果
                recognizer.Recognizing += Recognizer_Recognizing;
                // 收到最终结果
                recognizer.Recognized += Recognizer_Recognized;
                // 发生错误
                recognizer.Canceled += Recognizer_Canceled;

                // 启动语音识别器，开始持续监听音频输入
                recognizer.StartContinuousRecognitionAsync();

                // 设置意图预测器
                LUISRuntimeClient client = new LUISRuntimeClient(new ApiKeyServiceClientCredentials(luisKey));
                client.Endpoint = luisEndpoint;
                intentPrediction = new Prediction(client);
            }
            catch (Exception ex)
            {
                Log(ex.Message);
            }
        }

        // 识别过程中的中间结果
        private void Recognizer_Recognizing(object sender, SpeechRecognitionEventArgs e)
        {
            if (!string.IsNullOrEmpty(e.Result.Text))
            {
                Log("中间结果: " + e.Result.Text);
            }
        }

        // 出错时的处理
        private void Recognizer_Canceled(object sender, SpeechRecognitionCanceledEventArgs e)
        {
            Log("识别错误: " + e.ErrorDetails);
        }

        // 获得音频分析后的文本内容
        private void Recognizer_Recognized(object sender, SpeechRecognitionEventArgs e)
        {
            if (!string.IsNullOrEmpty(e.Result.Text))
            {
                Log("最终结果: " + e.Result.Text);
                ProcessSttResultAsync(e.Result.Text);
            }
        }

        private async void ProcessSttResultAsync(string text)
        {
            // 调用语言理解服务取得用户意图
            string intent = await GetIntentAsync(text);

            // 按照意图控制灯
            if (!string.IsNullOrEmpty(intent))
            {
                if (intent.Equals("TurnOn", StringComparison.OrdinalIgnoreCase))
                {
                    OpenLight();
                }
                else if (intent.Equals("TurnOff", StringComparison.OrdinalIgnoreCase))
                {
                    CloseLight();
                }
            }
        }

        private async Task<string> GetIntentAsync(string text)
        {
            try
            {
                var result = await intentPrediction.ResolveAsync(luisAppId, text);
                Log("意图: " + result.TopScoringIntent.Intent + "\r\n得分: " + result.TopScoringIntent.Score + "\r\n");
                return result.TopScoringIntent.Intent;
            }
            catch (Exception ex)
            {
                Log(ex.Message);
                return null;
            }
        }

        #region 界面操作

        private void Log(string message, params string[] parameters)
        {
            MakesureRunInUI(() =>
            {
                if (parameters != null && parameters.Length > 0)
                {
                    message = string.Format(message + "\r\n", parameters);
                }
                else
                {
                    message += "\r\n";
                }
                textBox1.AppendText(message);
            });
        }

        private void OpenLight()
        {
            MakesureRunInUI(() =>
            {
                pictureBox1.Load("LightOn.png");
            });
        }

        private void CloseLight()
        {
            MakesureRunInUI(() =>
            {
                pictureBox1.Load("LightOff.png");
            });
        }

        private void MakesureRunInUI(Action action)
        {
            if (InvokeRequired)
            {
                MethodInvoker method = new MethodInvoker(action);
                Invoke(action, null);
            }
            else
            {
                action();
            }
        }

        #endregion
    }
}