using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

// 这里是 Game 类需要暴露给外部组件的数据结构。我们希望这些数据结构尽量是不可修改的（immutable）。
// Immutable 类型方便我们辨明程序的逻辑，也易于多线程使用。
namespace GoldedNumberClient.Models
{
    /// <summary>
    /// 游戏房间对应的提交数字的模式。是提交一个数还是两个数。
    /// </summary>
    public enum RoomNumberMode
    {
        Unknown = 0,
        One = 1,
        Two = 2,
    }

    /// <summary>
    /// 表示新一轮提交。
    /// </summary>
    public class Round
    {
        public Round(string rid, RoomNumberMode mode)
        {
            RoundId = rid;
            NumberMode = mode;
        }

        public string RoundId { get; }

        public RoomNumberMode NumberMode { get; }
    }

    /// <summary>
    /// 新 <see cref="Game"/> 实例的创建结果。
    /// </summary>
    public class CreateGameResult
    {
        public CreateGameResult(Game game, TimeSpan cd, Round initRound)
        {
            Game = game;
            InitialCountdown = cd;
            InitialRound = initRound;
        }

        /// <summary>
        /// 获取创建好的 <see cref="Models.Game"/>实例。
        /// </summary>
        public Game Game { get; }

        /// <summary>
        /// 获取创建时这轮提交剩下的时间。
        /// </summary>
        public TimeSpan InitialCountdown { get; }

        /// <summary>
        /// 获取创建时正在进行的一轮提交。
        /// </summary>
        public Round InitialRound { get; }
    }

    /// <summary>
    /// 辅助类，用于在各种操作的结果类型上附加操作成功与否的信息。
    /// </summary>
    /// <typeparam name="T">要包装的操作结果类型。</typeparam>
    public class GameOperation<T>
    {
        private GameOperation() { }

        /// <summary>
        /// 操作成功了，将结果类型的实例包装起来。
        /// </summary>
        public static GameOperation<T> Succ(T result) => new GameOperation<T> { Succeeded = true, _result = result };

        /// <summary>
        /// 操作失败了，提供一些错误信息。
        /// </summary>
        public static GameOperation<T> Fail(string errorMsg) => new GameOperation<T> { Succeeded = false, _errMsg = errorMsg };

        /// <summary>
        /// 操作是否成功。
        /// </summary>
        public bool Succeeded { get; private set; }

        private T _result;
        private string _errMsg;

        /// <summary>
        /// 获取操作的结果。如果操作没有成功，将抛异常。
        /// </summary>
        public T OperationResult => Succeeded ? _result : throw new InvalidOperationException("Failed, no result.");

        /// <summary>
        /// 获取操作的错误信息。如果操作成功了，将抛异常。
        /// </summary>
        public string ErrorMessage => !Succeeded ? _errMsg : throw new InvalidOperationException("Succeeded, no error message.");
    }
}
