// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace MNIST.App
{
    partial class MainWindow
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
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(MainWindow));
            this.writeArea = new System.Windows.Forms.PictureBox();
            this.outputText = new System.Windows.Forms.Label();
            this.button1 = new System.Windows.Forms.Button();
            this.visualizeSwitch = new System.Windows.Forms.CheckBox();
            ((System.ComponentModel.ISupportInitialize)(this.writeArea)).BeginInit();
            this.SuspendLayout();
            // 
            // writeArea
            // 
            resources.ApplyResources(this.writeArea, "writeArea");
            this.writeArea.Name = "writeArea";
            this.writeArea.TabStop = false;
            this.writeArea.MouseDown += new System.Windows.Forms.MouseEventHandler(this.writeArea_MouseDown);
            this.writeArea.MouseMove += new System.Windows.Forms.MouseEventHandler(this.writeArea_MouseMove);
            this.writeArea.MouseUp += new System.Windows.Forms.MouseEventHandler(this.writeArea_MouseUp);
            // 
            // outputText
            // 
            resources.ApplyResources(this.outputText, "outputText");
            this.outputText.Name = "outputText";
            // 
            // button1
            // 
            resources.ApplyResources(this.button1, "button1");
            this.button1.Name = "button1";
            this.button1.UseVisualStyleBackColor = true;
            this.button1.Click += new System.EventHandler(this.clean_click);
            // 
            // visualizeSwitch
            // 
            resources.ApplyResources(this.visualizeSwitch, "visualizeSwitch");
            this.visualizeSwitch.Name = "visualizeSwitch";
            this.visualizeSwitch.UseVisualStyleBackColor = true;
            // 
            // MainWindow
            // 
            resources.ApplyResources(this, "$this");
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.Controls.Add(this.visualizeSwitch);
            this.Controls.Add(this.button1);
            this.Controls.Add(this.outputText);
            this.Controls.Add(this.writeArea);
            this.Name = "MainWindow";
            this.Load += new System.EventHandler(this.Form1_Load);
            ((System.ComponentModel.ISupportInitialize)(this.writeArea)).EndInit();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.PictureBox writeArea;
        private System.Windows.Forms.Label outputText;
        private System.Windows.Forms.Button button1;
        private System.Windows.Forms.CheckBox visualizeSwitch;
    }
}