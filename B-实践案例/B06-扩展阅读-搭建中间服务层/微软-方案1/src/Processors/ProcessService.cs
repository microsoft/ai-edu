// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using CognitiveMiddlewareService.MiddlewareService;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace CognitiveMiddlewareService.Processors
{
    public class ProcessService : IProcessService
    {
        private readonly ILandmarkService landmarkService;
        private readonly ICelebrityService celebrityService;

        public ProcessService(ILandmarkService ls, ICelebrityService cs)
        {
            this.landmarkService = ls;
            this.celebrityService = cs;
        }

        public async Task<AggregatedResult> Process(byte[] imgData)
        {
            // preprocess
            // todo: create screening image classifier to get a rough category, then decide call which service

            // task dispatcher: parallelized run 'Do'
            // todo: put this logic into Dispatcher service
            List<Task> listTask = new List<Task>();

            var taskLandmark = this.landmarkService.Do(imgData);
            listTask.Add(taskLandmark);
            var taskCelebrity = this.celebrityService.Do(imgData);
            listTask.Add(taskCelebrity);
            await Task.WhenAll(listTask);
            LandmarkResult lmResult = taskLandmark.Result;
            CelebrityResult cbResult = taskCelebrity.Result;

            // aggregator
            // todo: put this logic into Aggregator service
            AggregatedResult ar = new AggregatedResult()
            {
                Landmark = lmResult,
                Celebrity = cbResult
            };

            return ar;
  

            // ranker
            // todo: if there have more than one result in AgregatedResult, need give them a ranking

            // output generator
            // todo: generate specified JSON data, such as Adptive Card
        }
    }
}