// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;

namespace GoldedNumberClient.Utils
{
    /// <summary>
    /// 一些 XAML 辅助类。
    /// </summary>
    public class AllChildren : DependencyObject
    {
        #region Margin
        public static Thickness GetMargin(DependencyObject obj)
        {
            return (Thickness)obj.GetValue(MarginProperty);
        }

        public static void SetMargin(DependencyObject obj, Thickness value)
        {
            obj.SetValue(MarginProperty, value);
        }

        public static readonly DependencyProperty MarginProperty =
            DependencyProperty.RegisterAttached("Margin", typeof(Thickness), typeof(AllChildren), new PropertyMetadata(AllChildrenMarginChangedCallback));

        // 这个附加依赖属性的效果，是无视XAML容器中子元素的类型，统一地为子元素设置 Margin（如果该元素的Margin没有被单独设置过的话）。
        public static void AllChildrenMarginChangedCallback(DependencyObject d, DependencyPropertyChangedEventArgs e)
        {
            if (d is Panel p)
            {
                p.Initialized += (_1, _2) =>
                {
                    var margin = GetMargin(p);
                    foreach (var c in p.Children)
                    {
                        if (c is FrameworkElement fe)
                        {
                            // 检查是否有更高优先级的设置。没有的话我们才应用统一设置。
                            ValueSource vs = DependencyPropertyHelper.GetValueSource(fe, FrameworkElement.MarginProperty);
                            if (vs.BaseValueSource < BaseValueSource.Style)
                            {
                                fe.Margin = margin;
                            }
                        }
                    }
                };
            }
        }
        #endregion // !Margin
    }
}