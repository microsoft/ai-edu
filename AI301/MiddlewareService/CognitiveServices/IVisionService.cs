using System.IO;
using System.Threading.Tasks;

namespace CognitiveMiddlewareService.CognitiveServices
{
    public interface IVisionService
    {
        Task<Landmark> RecognizeLandmarkAsync(Stream imgStream);

        Task<Celebrity> RecognizeCelebrityAsync(Stream imgStream);
    }
}
