// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace CognitiveMiddlewareService.CognitiveServices
{
    public class EntityResult
    {
        public string _type { get; set; }
        public Querycontext queryContext { get; set; }
        public Entities entities { get; set; }
        public Rankingresponse rankingResponse { get; set; }
    }

    public class Querycontext
    {
        public string originalQuery { get; set; }
    }

    public class Entities
    {
        public Value[] value { get; set; }
    }

    public class Value
    {
        public string id { get; set; }
        public Contractualrule[] contractualRules { get; set; }
        public string webSearchUrl { get; set; }
        public string name { get; set; }
        public string url { get; set; }
        public Image image { get; set; }
        public string description { get; set; }
        public Entitypresentationinfo entityPresentationInfo { get; set; }
        public string bingId { get; set; }
    }

    public class Image
    {
        public string name { get; set; }
        public string thumbnailUrl { get; set; }
        public Provider[] provider { get; set; }
        public string hostPageUrl { get; set; }
        public int width { get; set; }
        public int height { get; set; }
        public int sourceWidth { get; set; }
        public int sourceHeight { get; set; }
    }

    public class Provider
    {
        public string _type { get; set; }
        public string url { get; set; }
    }

    public class Entitypresentationinfo
    {
        public string entityScenario { get; set; }
        public string[] entityTypeHints { get; set; }
    }

    public class Contractualrule
    {
        public string _type { get; set; }
        public string targetPropertyName { get; set; }
        public bool mustBeCloseToContent { get; set; }
        public License license { get; set; }
        public string licenseNotice { get; set; }
        public string text { get; set; }
        public string url { get; set; }
    }

    public class License
    {
        public string name { get; set; }
        public string url { get; set; }
    }

    public class Rankingresponse
    {
        public Sidebar sidebar { get; set; }
    }

    public class Sidebar
    {
        public Item[] items { get; set; }
    }

    public class Item
    {
        public string answerType { get; set; }
        public int resultIndex { get; set; }
        public Value1 value { get; set; }
    }

    public class Value1
    {
        public string id { get; set; }
    }

}