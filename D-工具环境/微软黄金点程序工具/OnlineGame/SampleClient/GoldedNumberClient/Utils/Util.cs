// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GoldedNumberClient.Utils
{
    public static class Util
    {
        /// <summary>
        /// 安全地触发一个<see cref="TypedEventHandler{S, A}"/>类型的事件。
        /// 原理是，如果一个事件没有绑定任何处理器，访问该事件得到的是 null，容易导致 <see cref="NullReferenceException"/>。
        /// </summary>
        /// <typeparam name="S">发送事件的类型。</typeparam>
        /// <typeparam name="A">事件的参数类型。</typeparam>
        /// <param name="evt">事件对象。</param>
        /// <param name="action">如何触发该事件。</param>
        public static void SafeInvoke<S, A>(this TypedEventHandler<S, A> evt, Action<TypedEventHandler<S, A>> action)
        {
            // evt 是按值拷贝的，不会出现多线程下，第二次读取就是 null 的情况。
            if (evt != null)
            {
                // 我们通过这种抽象，非常容易实现除了直接通过参数调用以外的触发方式，比如在后台线程触发。
                action(evt);
            }
        }
    }

    /// <summary>
    /// 泛型的委托类型。
    /// </summary>
    /// <typeparam name="S">发送者的类型。</typeparam>
    /// <typeparam name="A">委托参数的类型。</typeparam>
    /// <param name="sender">发送者对象。</param>
    /// <param name="arg">参数对象。</param>
    public delegate void TypedEventHandler<S, A>(S sender, A arg);
}