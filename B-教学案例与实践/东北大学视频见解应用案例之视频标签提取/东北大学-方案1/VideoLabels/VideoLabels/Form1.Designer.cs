namespace VideoLabels
{
    partial class VideoLabel
    {
        /// <summary>
        /// 必需的设计器变量。
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// 清理所有正在使用的资源。
        /// </summary>
        /// <param name="disposing">如果应释放托管资源，为 true；否则为 false。</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows 窗体设计器生成的代码

        /// <summary>
        /// 设计器支持所需的方法 - 不要修改
        /// 使用代码编辑器修改此方法的内容。
        /// </summary>
        private void InitializeComponent()
        {
            this.button_Select = new System.Windows.Forms.Button();
            this.textBox_Status = new System.Windows.Forms.TextBox();
            this.textBox_FilePath = new System.Windows.Forms.TextBox();
            this.textBox_VideoLabels = new System.Windows.Forms.TextBox();
            this.labelStatus = new System.Windows.Forms.Label();
            this.labelVideoLabel = new System.Windows.Forms.Label();
            this.button_Upload = new System.Windows.Forms.Button();
            this.button_GetVideosID = new System.Windows.Forms.Button();
            this.labelListofVideoID = new System.Windows.Forms.Label();
            this.textBox_VideoIDList = new System.Windows.Forms.TextBox();
            this.labelInputVideoID = new System.Windows.Forms.Label();
            this.textBox_VideoID = new System.Windows.Forms.TextBox();
            this.button_GetLabelsWithID = new System.Windows.Forms.Button();
            this.label1 = new System.Windows.Forms.Label();
            this.textBox_AccountID = new System.Windows.Forms.TextBox();
            this.label2 = new System.Windows.Forms.Label();
            this.textBox_PrimaryKey = new System.Windows.Forms.TextBox();
            this.button_Setting = new System.Windows.Forms.Button();
            this.SuspendLayout();
            // 
            // button_Select
            // 
            this.button_Select.Location = new System.Drawing.Point(42, 89);
            this.button_Select.Name = "button_Select";
            this.button_Select.Size = new System.Drawing.Size(115, 23);
            this.button_Select.TabIndex = 1;
            this.button_Select.Text = "选择视频：";
            this.button_Select.UseVisualStyleBackColor = true;
            this.button_Select.Click += new System.EventHandler(this.Button_Select_Click);
            // 
            // textBox_Status
            // 
            this.textBox_Status.Location = new System.Drawing.Point(46, 198);
            this.textBox_Status.Multiline = true;
            this.textBox_Status.Name = "textBox_Status";
            this.textBox_Status.ScrollBars = System.Windows.Forms.ScrollBars.Vertical;
            this.textBox_Status.Size = new System.Drawing.Size(279, 326);
            this.textBox_Status.TabIndex = 2;
            // 
            // textBox_FilePath
            // 
            this.textBox_FilePath.Location = new System.Drawing.Point(210, 90);
            this.textBox_FilePath.Name = "textBox_FilePath";
            this.textBox_FilePath.Size = new System.Drawing.Size(393, 21);
            this.textBox_FilePath.TabIndex = 3;
            // 
            // textBox_VideoLabels
            // 
            this.textBox_VideoLabels.Location = new System.Drawing.Point(550, 198);
            this.textBox_VideoLabels.Multiline = true;
            this.textBox_VideoLabels.Name = "textBox_VideoLabels";
            this.textBox_VideoLabels.ScrollBars = System.Windows.Forms.ScrollBars.Vertical;
            this.textBox_VideoLabels.Size = new System.Drawing.Size(195, 326);
            this.textBox_VideoLabels.TabIndex = 4;
            // 
            // labelStatus
            // 
            this.labelStatus.AutoSize = true;
            this.labelStatus.Location = new System.Drawing.Point(46, 183);
            this.labelStatus.Name = "labelStatus";
            this.labelStatus.Size = new System.Drawing.Size(65, 12);
            this.labelStatus.TabIndex = 5;
            this.labelStatus.Text = "运行状态：";
            // 
            // labelVideoLabel
            // 
            this.labelVideoLabel.AutoSize = true;
            this.labelVideoLabel.Location = new System.Drawing.Point(548, 183);
            this.labelVideoLabel.Name = "labelVideoLabel";
            this.labelVideoLabel.Size = new System.Drawing.Size(65, 12);
            this.labelVideoLabel.TabIndex = 6;
            this.labelVideoLabel.Text = "视频标签：";
            // 
            // button_Upload
            // 
            this.button_Upload.Location = new System.Drawing.Point(643, 89);
            this.button_Upload.Name = "button_Upload";
            this.button_Upload.Size = new System.Drawing.Size(98, 23);
            this.button_Upload.TabIndex = 7;
            this.button_Upload.Text = "上传并检索";
            this.button_Upload.UseVisualStyleBackColor = true;
            this.button_Upload.Click += new System.EventHandler(this.Button_Upload_Click);
            // 
            // button_GetVideosID
            // 
            this.button_GetVideosID.Location = new System.Drawing.Point(42, 132);
            this.button_GetVideosID.Name = "button_GetVideosID";
            this.button_GetVideosID.Size = new System.Drawing.Size(115, 23);
            this.button_GetVideosID.TabIndex = 8;
            this.button_GetVideosID.Text = "获取视频列表";
            this.button_GetVideosID.UseVisualStyleBackColor = true;
            this.button_GetVideosID.Click += new System.EventHandler(this.Button_GetVideosID_Click);
            // 
            // labelListofVideoID
            // 
            this.labelListofVideoID.AutoSize = true;
            this.labelListofVideoID.Location = new System.Drawing.Point(338, 183);
            this.labelListofVideoID.Name = "labelListofVideoID";
            this.labelListofVideoID.Size = new System.Drawing.Size(77, 12);
            this.labelListofVideoID.TabIndex = 11;
            this.labelListofVideoID.Text = "视频ID列表：";
            // 
            // textBox_VideoIDList
            // 
            this.textBox_VideoIDList.Location = new System.Drawing.Point(340, 198);
            this.textBox_VideoIDList.Multiline = true;
            this.textBox_VideoIDList.Name = "textBox_VideoIDList";
            this.textBox_VideoIDList.ScrollBars = System.Windows.Forms.ScrollBars.Vertical;
            this.textBox_VideoIDList.Size = new System.Drawing.Size(195, 326);
            this.textBox_VideoIDList.TabIndex = 12;
            // 
            // labelInputVideoID
            // 
            this.labelInputVideoID.AutoSize = true;
            this.labelInputVideoID.Location = new System.Drawing.Point(208, 137);
            this.labelInputVideoID.Name = "labelInputVideoID";
            this.labelInputVideoID.Size = new System.Drawing.Size(77, 12);
            this.labelInputVideoID.TabIndex = 13;
            this.labelInputVideoID.Text = "输入视频ID：";
            // 
            // textBox_VideoID
            // 
            this.textBox_VideoID.Location = new System.Drawing.Point(291, 134);
            this.textBox_VideoID.Name = "textBox_VideoID";
            this.textBox_VideoID.Size = new System.Drawing.Size(312, 21);
            this.textBox_VideoID.TabIndex = 14;
            // 
            // button_GetLabelsWithID
            // 
            this.button_GetLabelsWithID.Location = new System.Drawing.Point(643, 134);
            this.button_GetLabelsWithID.Name = "button_GetLabelsWithID";
            this.button_GetLabelsWithID.Size = new System.Drawing.Size(98, 23);
            this.button_GetLabelsWithID.TabIndex = 15;
            this.button_GetLabelsWithID.Text = "检索视频标签";
            this.button_GetLabelsWithID.UseVisualStyleBackColor = true;
            this.button_GetLabelsWithID.Click += new System.EventHandler(this.Button_GetLabelsWithID_Click);
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(50, 23);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(113, 12);
            this.label1.TabIndex = 16;
            this.label1.Text = "输入 Account ID ：";
            // 
            // textBox_AccountID
            // 
            this.textBox_AccountID.Location = new System.Drawing.Point(210, 20);
            this.textBox_AccountID.Name = "textBox_AccountID";
            this.textBox_AccountID.Size = new System.Drawing.Size(393, 21);
            this.textBox_AccountID.TabIndex = 17;
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Location = new System.Drawing.Point(44, 50);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(119, 12);
            this.label2.TabIndex = 16;
            this.label2.Text = "输入 Primary Key ：";
            // 
            // textBox_PrimaryKey
            // 
            this.textBox_PrimaryKey.Location = new System.Drawing.Point(210, 47);
            this.textBox_PrimaryKey.Name = "textBox_PrimaryKey";
            this.textBox_PrimaryKey.Size = new System.Drawing.Size(393, 21);
            this.textBox_PrimaryKey.TabIndex = 17;
            // 
            // button_Setting
            // 
            this.button_Setting.Location = new System.Drawing.Point(643, 20);
            this.button_Setting.Name = "button_Setting";
            this.button_Setting.Size = new System.Drawing.Size(98, 48);
            this.button_Setting.TabIndex = 18;
            this.button_Setting.Text = "设定并初始化";
            this.button_Setting.UseVisualStyleBackColor = true;
            this.button_Setting.Click += new System.EventHandler(this.Button_Setting_Click);
            // 
            // VideoLabel
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 12F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(793, 536);
            this.Controls.Add(this.button_Setting);
            this.Controls.Add(this.textBox_PrimaryKey);
            this.Controls.Add(this.label2);
            this.Controls.Add(this.textBox_AccountID);
            this.Controls.Add(this.label1);
            this.Controls.Add(this.button_GetLabelsWithID);
            this.Controls.Add(this.textBox_VideoID);
            this.Controls.Add(this.labelInputVideoID);
            this.Controls.Add(this.textBox_VideoIDList);
            this.Controls.Add(this.labelListofVideoID);
            this.Controls.Add(this.button_GetVideosID);
            this.Controls.Add(this.button_Upload);
            this.Controls.Add(this.labelVideoLabel);
            this.Controls.Add(this.labelStatus);
            this.Controls.Add(this.textBox_VideoLabels);
            this.Controls.Add(this.textBox_FilePath);
            this.Controls.Add(this.textBox_Status);
            this.Controls.Add(this.button_Select);
            this.Name = "VideoLabel";
            this.StartPosition = System.Windows.Forms.FormStartPosition.CenterScreen;
            this.Text = "Video Label";
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion
        private System.Windows.Forms.Button button_Select;
        private System.Windows.Forms.TextBox textBox_Status;
        private System.Windows.Forms.TextBox textBox_FilePath;
        private System.Windows.Forms.TextBox textBox_VideoLabels;
        private System.Windows.Forms.Label labelStatus;
        private System.Windows.Forms.Label labelVideoLabel;
        private System.Windows.Forms.Button button_Upload;
        private System.Windows.Forms.Button button_GetVideosID;
        private System.Windows.Forms.Label labelListofVideoID;
        private System.Windows.Forms.TextBox textBox_VideoIDList;
        private System.Windows.Forms.Label labelInputVideoID;
        private System.Windows.Forms.TextBox textBox_VideoID;
        private System.Windows.Forms.Button button_GetLabelsWithID;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.TextBox textBox_AccountID;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.TextBox textBox_PrimaryKey;
        private System.Windows.Forms.Button button_Setting;
    }
}

