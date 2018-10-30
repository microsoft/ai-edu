namespace LocalManager
{
    partial class MainForm
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            System.Windows.Forms.DataVisualization.Charting.ChartArea chartArea3 = new System.Windows.Forms.DataVisualization.Charting.ChartArea();
            System.Windows.Forms.DataVisualization.Charting.ChartArea chartArea4 = new System.Windows.Forms.DataVisualization.Charting.ChartArea();
            this.historyView = new System.Windows.Forms.DataGridView();
            this.tbLog = new System.Windows.Forms.TextBox();
            this.btnSelectBots = new System.Windows.Forms.Button();
            this.label1 = new System.Windows.Forms.Label();
            this.tbRounds = new System.Windows.Forms.TextBox();
            this.btnPlayMultiRound = new System.Windows.Forms.Button();
            this.panel1 = new System.Windows.Forms.Panel();
            this.btnStop = new System.Windows.Forms.Button();
            this.btnExport = new System.Windows.Forms.Button();
            this.panel2 = new System.Windows.Forms.Panel();
            this.panel12 = new System.Windows.Forms.Panel();
            this.panel5 = new System.Windows.Forms.Panel();
            this.panel7 = new System.Windows.Forms.Panel();
            this.panel6 = new System.Windows.Forms.Panel();
            this.label3 = new System.Windows.Forms.Label();
            this.panel9 = new System.Windows.Forms.Panel();
            this.panel11 = new System.Windows.Forms.Panel();
            this.panel10 = new System.Windows.Forms.Panel();
            this.label4 = new System.Windows.Forms.Label();
            this.panel3 = new System.Windows.Forms.Panel();
            this.panel8 = new System.Windows.Forms.Panel();
            this.scoreView = new System.Windows.Forms.DataGridView();
            this.panel4 = new System.Windows.Forms.Panel();
            this.label2 = new System.Windows.Forms.Label();
            this.tableLayoutPanel1 = new System.Windows.Forms.TableLayoutPanel();
            this.panel13 = new System.Windows.Forms.Panel();
            this.chartScore = new System.Windows.Forms.DataVisualization.Charting.Chart();
            this.panel14 = new System.Windows.Forms.Panel();
            this.chartGoldenNumber = new System.Windows.Forms.DataVisualization.Charting.Chart();
            ((System.ComponentModel.ISupportInitialize)(this.historyView)).BeginInit();
            this.panel1.SuspendLayout();
            this.panel2.SuspendLayout();
            this.panel12.SuspendLayout();
            this.panel5.SuspendLayout();
            this.panel7.SuspendLayout();
            this.panel6.SuspendLayout();
            this.panel9.SuspendLayout();
            this.panel11.SuspendLayout();
            this.panel10.SuspendLayout();
            this.panel3.SuspendLayout();
            this.panel8.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.scoreView)).BeginInit();
            this.panel4.SuspendLayout();
            this.tableLayoutPanel1.SuspendLayout();
            this.panel13.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.chartScore)).BeginInit();
            this.panel14.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.chartGoldenNumber)).BeginInit();
            this.SuspendLayout();
            // 
            // historyView
            // 
            this.historyView.AllowUserToAddRows = false;
            this.historyView.AllowUserToDeleteRows = false;
            this.historyView.ColumnHeadersHeightSizeMode = System.Windows.Forms.DataGridViewColumnHeadersHeightSizeMode.AutoSize;
            this.historyView.Dock = System.Windows.Forms.DockStyle.Fill;
            this.historyView.Location = new System.Drawing.Point(0, 0);
            this.historyView.Name = "historyView";
            this.historyView.ReadOnly = true;
            this.historyView.RowTemplate.Height = 23;
            this.historyView.Size = new System.Drawing.Size(580, 414);
            this.historyView.TabIndex = 6;
            // 
            // tbLog
            // 
            this.tbLog.Dock = System.Windows.Forms.DockStyle.Fill;
            this.tbLog.Location = new System.Drawing.Point(0, 0);
            this.tbLog.Multiline = true;
            this.tbLog.Name = "tbLog";
            this.tbLog.ReadOnly = true;
            this.tbLog.ScrollBars = System.Windows.Forms.ScrollBars.Both;
            this.tbLog.Size = new System.Drawing.Size(271, 414);
            this.tbLog.TabIndex = 5;
            // 
            // btnSelectBots
            // 
            this.btnSelectBots.Location = new System.Drawing.Point(12, 4);
            this.btnSelectBots.Name = "btnSelectBots";
            this.btnSelectBots.Size = new System.Drawing.Size(136, 23);
            this.btnSelectBots.TabIndex = 4;
            this.btnSelectBots.Text = "选择Bots所在目录";
            this.btnSelectBots.UseVisualStyleBackColor = true;
            this.btnSelectBots.Click += new System.EventHandler(this.btnSelectBots_Click);
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(244, 9);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(53, 12);
            this.label1.TabIndex = 8;
            this.label1.Text = "回合数：";
            // 
            // tbRounds
            // 
            this.tbRounds.Location = new System.Drawing.Point(303, 5);
            this.tbRounds.Name = "tbRounds";
            this.tbRounds.Size = new System.Drawing.Size(86, 21);
            this.tbRounds.TabIndex = 9;
            this.tbRounds.Text = "1";
            // 
            // btnPlayMultiRound
            // 
            this.btnPlayMultiRound.Location = new System.Drawing.Point(392, 3);
            this.btnPlayMultiRound.Name = "btnPlayMultiRound";
            this.btnPlayMultiRound.Size = new System.Drawing.Size(54, 23);
            this.btnPlayMultiRound.TabIndex = 10;
            this.btnPlayMultiRound.Text = "开始";
            this.btnPlayMultiRound.UseVisualStyleBackColor = true;
            this.btnPlayMultiRound.Click += new System.EventHandler(this.btnPlayMultiRound_Click);
            // 
            // panel1
            // 
            this.panel1.Controls.Add(this.btnStop);
            this.panel1.Controls.Add(this.btnExport);
            this.panel1.Controls.Add(this.btnPlayMultiRound);
            this.panel1.Controls.Add(this.tbRounds);
            this.panel1.Controls.Add(this.label1);
            this.panel1.Controls.Add(this.btnSelectBots);
            this.panel1.Dock = System.Windows.Forms.DockStyle.Top;
            this.panel1.Location = new System.Drawing.Point(0, 0);
            this.panel1.Name = "panel1";
            this.panel1.Size = new System.Drawing.Size(1055, 29);
            this.panel1.TabIndex = 13;
            // 
            // btnStop
            // 
            this.btnStop.Location = new System.Drawing.Point(452, 3);
            this.btnStop.Name = "btnStop";
            this.btnStop.Size = new System.Drawing.Size(54, 23);
            this.btnStop.TabIndex = 16;
            this.btnStop.Text = "停止";
            this.btnStop.UseVisualStyleBackColor = true;
            this.btnStop.Click += new System.EventHandler(this.btnStop_Click);
            // 
            // btnExport
            // 
            this.btnExport.Location = new System.Drawing.Point(572, 3);
            this.btnExport.Name = "btnExport";
            this.btnExport.Size = new System.Drawing.Size(107, 23);
            this.btnExport.TabIndex = 15;
            this.btnExport.Text = "导出比赛数据";
            this.btnExport.UseVisualStyleBackColor = true;
            this.btnExport.Click += new System.EventHandler(this.btnExport_Click);
            // 
            // panel2
            // 
            this.panel2.Controls.Add(this.panel12);
            this.panel2.Controls.Add(this.panel9);
            this.panel2.Controls.Add(this.panel3);
            this.panel2.Controls.Add(this.tableLayoutPanel1);
            this.panel2.Dock = System.Windows.Forms.DockStyle.Fill;
            this.panel2.Location = new System.Drawing.Point(0, 29);
            this.panel2.Name = "panel2";
            this.panel2.Size = new System.Drawing.Size(1055, 632);
            this.panel2.TabIndex = 14;
            // 
            // panel12
            // 
            this.panel12.Controls.Add(this.panel5);
            this.panel12.Dock = System.Windows.Forms.DockStyle.Fill;
            this.panel12.Location = new System.Drawing.Point(200, 0);
            this.panel12.Name = "panel12";
            this.panel12.Size = new System.Drawing.Size(582, 449);
            this.panel12.TabIndex = 10;
            // 
            // panel5
            // 
            this.panel5.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.panel5.Controls.Add(this.panel7);
            this.panel5.Controls.Add(this.panel6);
            this.panel5.Dock = System.Windows.Forms.DockStyle.Fill;
            this.panel5.Location = new System.Drawing.Point(0, 0);
            this.panel5.Name = "panel5";
            this.panel5.Size = new System.Drawing.Size(582, 449);
            this.panel5.TabIndex = 9;
            // 
            // panel7
            // 
            this.panel7.Controls.Add(this.historyView);
            this.panel7.Dock = System.Windows.Forms.DockStyle.Fill;
            this.panel7.Location = new System.Drawing.Point(0, 33);
            this.panel7.Name = "panel7";
            this.panel7.Size = new System.Drawing.Size(580, 414);
            this.panel7.TabIndex = 7;
            // 
            // panel6
            // 
            this.panel6.Controls.Add(this.label3);
            this.panel6.Dock = System.Windows.Forms.DockStyle.Top;
            this.panel6.Location = new System.Drawing.Point(0, 0);
            this.panel6.Name = "panel6";
            this.panel6.Size = new System.Drawing.Size(580, 33);
            this.panel6.TabIndex = 0;
            // 
            // label3
            // 
            this.label3.AutoSize = true;
            this.label3.Location = new System.Drawing.Point(9, 15);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(53, 12);
            this.label3.TabIndex = 0;
            this.label3.Text = "比赛数据";
            // 
            // panel9
            // 
            this.panel9.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.panel9.Controls.Add(this.panel11);
            this.panel9.Controls.Add(this.panel10);
            this.panel9.Dock = System.Windows.Forms.DockStyle.Right;
            this.panel9.Location = new System.Drawing.Point(782, 0);
            this.panel9.Name = "panel9";
            this.panel9.Size = new System.Drawing.Size(273, 449);
            this.panel9.TabIndex = 7;
            // 
            // panel11
            // 
            this.panel11.Controls.Add(this.tbLog);
            this.panel11.Dock = System.Windows.Forms.DockStyle.Fill;
            this.panel11.Location = new System.Drawing.Point(0, 33);
            this.panel11.Name = "panel11";
            this.panel11.Size = new System.Drawing.Size(271, 414);
            this.panel11.TabIndex = 1;
            // 
            // panel10
            // 
            this.panel10.Controls.Add(this.label4);
            this.panel10.Dock = System.Windows.Forms.DockStyle.Top;
            this.panel10.Location = new System.Drawing.Point(0, 0);
            this.panel10.Name = "panel10";
            this.panel10.Size = new System.Drawing.Size(271, 33);
            this.panel10.TabIndex = 0;
            // 
            // label4
            // 
            this.label4.AutoSize = true;
            this.label4.Location = new System.Drawing.Point(10, 16);
            this.label4.Name = "label4";
            this.label4.Size = new System.Drawing.Size(29, 12);
            this.label4.TabIndex = 0;
            this.label4.Text = "日志";
            // 
            // panel3
            // 
            this.panel3.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.panel3.Controls.Add(this.panel8);
            this.panel3.Controls.Add(this.panel4);
            this.panel3.Dock = System.Windows.Forms.DockStyle.Left;
            this.panel3.Location = new System.Drawing.Point(0, 0);
            this.panel3.Name = "panel3";
            this.panel3.Size = new System.Drawing.Size(200, 449);
            this.panel3.TabIndex = 8;
            // 
            // panel8
            // 
            this.panel8.Controls.Add(this.scoreView);
            this.panel8.Dock = System.Windows.Forms.DockStyle.Fill;
            this.panel8.Location = new System.Drawing.Point(0, 33);
            this.panel8.Name = "panel8";
            this.panel8.Size = new System.Drawing.Size(198, 414);
            this.panel8.TabIndex = 10;
            // 
            // scoreView
            // 
            this.scoreView.AllowUserToAddRows = false;
            this.scoreView.AllowUserToDeleteRows = false;
            this.scoreView.ColumnHeadersHeightSizeMode = System.Windows.Forms.DataGridViewColumnHeadersHeightSizeMode.AutoSize;
            this.scoreView.Dock = System.Windows.Forms.DockStyle.Fill;
            this.scoreView.Location = new System.Drawing.Point(0, 0);
            this.scoreView.Name = "scoreView";
            this.scoreView.ReadOnly = true;
            this.scoreView.RowTemplate.Height = 23;
            this.scoreView.Size = new System.Drawing.Size(198, 414);
            this.scoreView.TabIndex = 7;
            // 
            // panel4
            // 
            this.panel4.Controls.Add(this.label2);
            this.panel4.Dock = System.Windows.Forms.DockStyle.Top;
            this.panel4.Location = new System.Drawing.Point(0, 0);
            this.panel4.Name = "panel4";
            this.panel4.Size = new System.Drawing.Size(198, 33);
            this.panel4.TabIndex = 9;
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Location = new System.Drawing.Point(10, 15);
            this.label2.Margin = new System.Windows.Forms.Padding(0);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(29, 12);
            this.label2.TabIndex = 8;
            this.label2.Text = "得分";
            // 
            // tableLayoutPanel1
            // 
            this.tableLayoutPanel1.ColumnCount = 2;
            this.tableLayoutPanel1.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Percent, 50F));
            this.tableLayoutPanel1.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Percent, 50F));
            this.tableLayoutPanel1.Controls.Add(this.panel13, 0, 0);
            this.tableLayoutPanel1.Controls.Add(this.panel14, 1, 0);
            this.tableLayoutPanel1.Dock = System.Windows.Forms.DockStyle.Bottom;
            this.tableLayoutPanel1.Location = new System.Drawing.Point(0, 449);
            this.tableLayoutPanel1.Name = "tableLayoutPanel1";
            this.tableLayoutPanel1.RowCount = 1;
            this.tableLayoutPanel1.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Percent, 50F));
            this.tableLayoutPanel1.Size = new System.Drawing.Size(1055, 183);
            this.tableLayoutPanel1.TabIndex = 11;
            // 
            // panel13
            // 
            this.panel13.Controls.Add(this.chartScore);
            this.panel13.Dock = System.Windows.Forms.DockStyle.Fill;
            this.panel13.Location = new System.Drawing.Point(3, 3);
            this.panel13.Name = "panel13";
            this.panel13.Size = new System.Drawing.Size(521, 177);
            this.panel13.TabIndex = 0;
            // 
            // chartScore
            // 
            chartArea3.Name = "ChartArea1";
            this.chartScore.ChartAreas.Add(chartArea3);
            this.chartScore.Dock = System.Windows.Forms.DockStyle.Fill;
            this.chartScore.Location = new System.Drawing.Point(0, 0);
            this.chartScore.Name = "chartScore";
            this.chartScore.Size = new System.Drawing.Size(521, 177);
            this.chartScore.TabIndex = 0;
            this.chartScore.Text = "chart1";
            // 
            // panel14
            // 
            this.panel14.Controls.Add(this.chartGoldenNumber);
            this.panel14.Dock = System.Windows.Forms.DockStyle.Fill;
            this.panel14.Location = new System.Drawing.Point(530, 3);
            this.panel14.Name = "panel14";
            this.panel14.Size = new System.Drawing.Size(522, 177);
            this.panel14.TabIndex = 1;
            // 
            // chartGoldenNumber
            // 
            chartArea4.Name = "ChartArea1";
            this.chartGoldenNumber.ChartAreas.Add(chartArea4);
            this.chartGoldenNumber.Dock = System.Windows.Forms.DockStyle.Fill;
            this.chartGoldenNumber.Location = new System.Drawing.Point(0, 0);
            this.chartGoldenNumber.Name = "chartGoldenNumber";
            this.chartGoldenNumber.Size = new System.Drawing.Size(522, 177);
            this.chartGoldenNumber.TabIndex = 0;
            this.chartGoldenNumber.Text = "chart2";
            // 
            // MainForm
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 12F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(1055, 661);
            this.Controls.Add(this.panel2);
            this.Controls.Add(this.panel1);
            this.Name = "MainForm";
            this.Text = "Local Manager";
            ((System.ComponentModel.ISupportInitialize)(this.historyView)).EndInit();
            this.panel1.ResumeLayout(false);
            this.panel1.PerformLayout();
            this.panel2.ResumeLayout(false);
            this.panel12.ResumeLayout(false);
            this.panel5.ResumeLayout(false);
            this.panel7.ResumeLayout(false);
            this.panel6.ResumeLayout(false);
            this.panel6.PerformLayout();
            this.panel9.ResumeLayout(false);
            this.panel11.ResumeLayout(false);
            this.panel11.PerformLayout();
            this.panel10.ResumeLayout(false);
            this.panel10.PerformLayout();
            this.panel3.ResumeLayout(false);
            this.panel8.ResumeLayout(false);
            ((System.ComponentModel.ISupportInitialize)(this.scoreView)).EndInit();
            this.panel4.ResumeLayout(false);
            this.panel4.PerformLayout();
            this.tableLayoutPanel1.ResumeLayout(false);
            this.panel13.ResumeLayout(false);
            ((System.ComponentModel.ISupportInitialize)(this.chartScore)).EndInit();
            this.panel14.ResumeLayout(false);
            ((System.ComponentModel.ISupportInitialize)(this.chartGoldenNumber)).EndInit();
            this.ResumeLayout(false);

        }

        #endregion
        private System.Windows.Forms.DataGridView historyView;
        private System.Windows.Forms.TextBox tbLog;
        private System.Windows.Forms.Button btnSelectBots;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.TextBox tbRounds;
        private System.Windows.Forms.Button btnPlayMultiRound;
        private System.Windows.Forms.Panel panel1;
        private System.Windows.Forms.Panel panel2;
        private System.Windows.Forms.Panel panel5;
        private System.Windows.Forms.Panel panel6;
        private System.Windows.Forms.Label label3;
        private System.Windows.Forms.Panel panel3;
        private System.Windows.Forms.Panel panel4;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.DataGridView scoreView;
        private System.Windows.Forms.Panel panel7;
        private System.Windows.Forms.Panel panel8;
        private System.Windows.Forms.Panel panel9;
        private System.Windows.Forms.Panel panel11;
        private System.Windows.Forms.Panel panel10;
        private System.Windows.Forms.Label label4;
        private System.Windows.Forms.Panel panel12;
        private System.Windows.Forms.Button btnExport;
        private System.Windows.Forms.Button btnStop;
        private System.Windows.Forms.TableLayoutPanel tableLayoutPanel1;
        private System.Windows.Forms.Panel panel13;
        private System.Windows.Forms.DataVisualization.Charting.Chart chartScore;
        private System.Windows.Forms.Panel panel14;
        private System.Windows.Forms.DataVisualization.Charting.Chart chartGoldenNumber;
    }
}