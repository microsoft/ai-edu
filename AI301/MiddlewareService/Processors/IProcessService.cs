// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Threading.Tasks;

namespace CognitiveMiddlewareService.Processors
{
    public interface IProcessService
    {
        Task<AggregatedResult> Process(byte[] imgData);
    }
}