// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;

namespace CartoonTranslate
{
    class Cluster
    {
        const double x_gate = 10d;
        const double y_gate = 10d;

        public List<Rect> listClusterRect = new List<Rect>();

        public List<Rect> Dok(List<Rect> listRect)
        {
            Rect clusterRect = CreateClusterRect(listRect[0]);
            listClusterRect.Add(clusterRect);

            for (int i = 1; i < listRect.Count; i++)
            {
                bool needNewCluster = true;
                for (int j = 0; j < listClusterRect.Count; j++)
                {
                    Rect cluRect = listClusterRect[j];
                    if (CheckDistance(cluRect, listRect[i]))
                    {
                        // combine
                        cluRect.Union(listRect[i]);
                        needNewCluster = false;
                        listClusterRect[j] = cluRect;
                        break;
                    }
                }
                // create new cluster
                if (needNewCluster)
                {
                    Rect newCluRect = CreateClusterRect(listRect[i]);
                    listClusterRect.Add(newCluRect);
                }
            }


            return listClusterRect;
        }

        private Rect CreateClusterRect(Rect rectOrigin)
        {
            Rect rect = new Rect()
            {
                X = rectOrigin.X,
                Y = rectOrigin.Y,
                Height = rectOrigin.Height,
                Width = rectOrigin.Width
            };
            return rect;
        }

        private bool CheckDistance(Rect rect1, Rect rect2)
        {
            double[] distance = Distance(rect1, rect2);

            if (distance[0] < x_gate)
            {
                if (distance[1] == 0)
                {
                    return true;
                }
                else if (distance[1] < y_gate)
                {
                    return true;
                }
            }

            if (distance[1] < y_gate)
            {
                if (distance[0] == 0)
                {
                    return true;
                }
                else if (distance[0] < x_gate)
                {
                    return true;
                }
            }

            return false;
        }


        /// <summary>
        /// check x and y distance
        /// </summary>
        /// <param name="rect1"></param>
        /// <param name="rect2"></param>
        /// <returns></returns>
        private double[] Distance(Rect rect1, Rect rect2)
        {
            double[] distance_x_y = new double[2];
            // rect1 is at left side of rect2
            if (rect1.Left < rect2.Left && rect1.Right < rect2.Left)
            {
                distance_x_y[0] = rect2.Left - rect1.Right;
            }
            // right
            else if (rect1.Left > rect2.Right && rect1.Right > rect2.Right)
            {
                distance_x_y[0] = rect1.Left - rect2.Right;
            }
            // middel
            else
            {
                distance_x_y[0] = 0;
            }


            if (rect1.Top < rect2.Top && rect1.Bottom < rect2.Top)
            {
                distance_x_y[1] = rect2.Top - rect1.Bottom;
            }
            else if (rect1.Top > rect2.Bottom && rect1.Bottom > rect2.Bottom)
            {
                distance_x_y[1] = rect1.Top - rect2.Bottom;
            }
            else
            {
                distance_x_y[1] = 0;
            }

            return distance_x_y;
        }



    }
}