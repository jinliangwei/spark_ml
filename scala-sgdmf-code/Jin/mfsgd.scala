import scala.io.Source
import scala.collection.mutable.ListBuffer
import scala.collection.mutable.ArrayBuffer
import Array._
import scala.util.Random
import org.apache.commons.math3.linear._
import org.apache.spark._
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV}
import org.apache.spark.storage.StorageLevel

object SingleSgdmf{
  val negativeFactors = false
  def get_block_dim(num_ids:Int, num_blocks:Int, block_id:Int):Int = {
    var cut_off_block = num_ids % num_blocks
    var block_dim = 0
    if (block_id < cut_off_block){
        block_dim = num_ids / num_blocks + 1
    }else{
        block_dim = num_ids / num_blocks
    }
    block_dim
  }

  def randomMatrix(m: Int, n: Int) = {
    // change this after... block number or anything for seed value 
    val rand = new Random(42)
    if (negativeFactors) {
      new BDM(m, n, Array.fill(m * n)(rand.nextDouble() * 2 - 1))
    } else {
      new BDM(m, n, Array.fill(m * n)(rand.nextDouble()))
    }
  }

  def arraydot(a: Array[Double], b: Array[Double]) = {
    ((a zip b).map{case (x,y) => x*y} :\ 0.0){_ + _}
  }

  def get_block_id(num_ids:Int, num_blocks:Int, my_id:Int):Int = {
    var cut_off_block = num_ids % num_blocks
    var cut_off_block_offset = (num_ids / num_blocks + 1) * cut_off_block
    var block_id = 0
    if(my_id >= cut_off_block_offset)
      block_id = (my_id - cut_off_block_offset) / (num_ids / num_blocks) + cut_off_block
    else
      block_id = my_id / (num_ids / num_blocks + 1)
    block_id
  }

  def makepart(in:Int, ite:Iterator[(Int, (Int, Int, Float))], num_y_ids:Int,  TN:Int):Iterator[(Int, Array[Array[(Int, Int, Float)]])] = {
    var ar_buff = ArrayBuffer[ArrayBuffer[(Int, Int, Float)]]()
    for(a <- 0 until TN ){
      ar_buff += ArrayBuffer[(Int, Int, Float)]()
    }
    for (elem <- ite){
      val ybid = get_block_id(num_y_ids, TN, elem._2._2)
      ar_buff(ybid) += Tuple3(elem._2._1, elem._2._2, elem._2._3)
      assert(in == elem._1, "Index does not match")
    }
    val arbuff_array = ar_buff.map(a => a.toArray).toArray
    val tmp = Tuple2(in, arbuff_array)
    val tmpar = Array(tmp)
    tmpar.iterator
  }

  def PrintSetting(data_path:String, N:Int, rank:Int):Unit = {
    println("Setting Configuration Data Path (%s) N(%d) R(%d)\n".format(data_path, N, rank))
  }

  def generate_init(blockid:Int, rank:Int, blockdim:Int) = {
    val initblk = randomMatrix(blockdim, rank)
    Tuple2(blockid, initblk)
  }

  def get_block_offset(num_ids:Int, num_blocks:Int, block_id:Int):Int = {
    val cut_off_block = num_ids % num_blocks
    var block_offset = 0
    if (block_id <= cut_off_block){
      block_offset = (num_ids / num_blocks + 1) * block_id
    }else{
      block_offset = (num_ids / num_blocks + 1) * cut_off_block + (num_ids / num_blocks) * (block_id - cut_off_block)
    }
    block_offset
  }

  def compute_a_block(ite:Iterator[((Int, ((Array[Array[(Int, Int, Float)]], breeze.linalg.DenseMatrix[Double]), breeze.linalg.DenseMatrix[Double])))], subepoch:Int, num_x_ids:Int, num_y_ids:Int, N:Int, stepsize:Double) = {
    var arbuffer = ArrayBuffer[Tuple2[Int, ((Array[Array[(Int, Int, Float)]], breeze.linalg.DenseMatrix[Double]), breeze.linalg.DenseMatrix[Double])]]()
    var count = 0 
    var itercnt = 0
    var count_per_iter=0
    for (elem <- ite){
      itercnt += 1
      val value = elem._2;
      var w = value._1._2
      var h = value._2
      w = w.t
      h = h.t
      val block_x_id = elem._1
      val block_y_id = (block_x_id + subepoch) % N
      val cblk = value._1._1(block_y_id)
      val x_id_offset = get_block_offset(num_x_ids, N, block_x_id)
      val y_id_offset = get_block_offset(num_y_ids, N, block_y_id)
      count_per_iter=0
      for( rate <- cblk ){
        count_per_iter += 1
        count += 1
        val x_id = rate._1
        val y_id = rate._2
        val score:Double = rate._3
        val w_id = x_id - x_id_offset
        val h_id = y_id - y_id_offset
        val dp:Double = w(::,w_id) dot h(::,h_id)
        val diff:Double = score - dp       
        val w_gradient:BDV[Double] = h(::,h_id)*diff*(-2.0)
        val neww = w(::,w_id) - w_gradient*stepsize
        w(::, w_id) := neww
        val h_gradient:BDV[Double] = w(::,w_id)*diff*(-2.0)
        val newh = h(::,h_id) - h_gradient*stepsize
        h(::,h_id) := newh
      }
      val tmp = Tuple2(elem._1, ((value._1._1, w.t), h.t))
      arbuffer += tmp
    }
    val ret = arbuffer.toArray
    ret.iterator
  }

  def wrapper(ite:Iterator[((Int, ((Array[Array[(Int, Int, Float)]], breeze.linalg.DenseMatrix[Double]), breeze.linalg.DenseMatrix[Double])))]) = {
    var arbuffer = ArrayBuffer[Tuple2[Int, (Array[Array[(Int, Int, Float)]], breeze.linalg.DenseMatrix[Double])]]()
    for (elem <- ite){
      val tmp = Tuple2(elem._1, (elem._2._1._1, elem._2._1._2))
      arbuffer += tmp
    }
    val ret = arbuffer.toArray
    ret.iterator
  }

  def update_eval_h_ids(ite:Iterator[((Int, breeze.linalg.DenseMatrix[Double]))], subepoch:Int, N:Int) = {
    var arbuffer = ArrayBuffer[(Int, breeze.linalg.DenseMatrix[Double])]()
    for (elem <- ite){
      var newidx = (elem._1 + N - subepoch) % N
      val tmp = Tuple2(newidx, elem._2)
      arbuffer += tmp
    }
    val ret = arbuffer.toArray
    ret.iterator
  }

  def update_h_ids(ite:Iterator[((Int, (Array[Array[(Int, Int, Float)]], breeze.linalg.DenseMatrix[Double], breeze.linalg.DenseMatrix[Double])))], N:Int) = {
    var arbuffer = ArrayBuffer[(Int, breeze.linalg.DenseMatrix[Double])]()
    for (elem <- ite){
      var newidx = (elem._1 + N - 1) % N 
      val tmp = Tuple2(newidx, elem._2._3)
      arbuffer += tmp
    }
    val ret = arbuffer.toArray
    ret.iterator
  }

  def update_h_ids_onlymap(elem:(Int, ((Array[Array[(Int, Int, Float)]], breeze.linalg.DenseMatrix[Double]), breeze.linalg.DenseMatrix[Double])), N:Int) = {
    var newidx = (elem._1 + N - 1) % N
    Tuple2(newidx, elem._2._2)
  }

  def update_eval_h_ids_onlymap(elem:((Int, breeze.linalg.DenseMatrix[Double])), subepoch:Int, N:Int) = {
    var newidx = (elem._1 + N - subepoch) % N
    Tuple2(newidx, elem._2)
  }

  def eval_a_block(elem:(Int, ((Array[Array[(Int, Int, Float)]], breeze.linalg.DenseMatrix[Double]), breeze.linalg.DenseMatrix[Double])), subepoch:Int, num_x_ids:Int, num_y_ids:Int, N:Int, erroracc:org.apache.spark.Accumulator[Double], countacc:org.apache.spark.Accumulator[Int]) = {
    var err:Double=0.0
    var count:Int = 0
    val value = elem._2;
    var w = value._1._2
    var h = value._2
    w = w.t
    h = h.t
    val block_x_id = elem._1
    val block_y_id = (block_x_id + subepoch) % N
    val cblk = value._1._1(block_y_id)
    val x_id_offset = get_block_offset(num_x_ids, N, block_x_id)
    val y_id_offset = get_block_offset(num_y_ids, N, block_y_id)
    for( rate <- cblk ){
      count += 1
      val x_id = rate._1
      val y_id = rate._2
      val score:Double = rate._3
      val w_id = x_id - x_id_offset
      val h_id = y_id - y_id_offset
      val dp:Double = w(::,w_id) dot h(::,h_id)
      val diff:Double = score - dp
      err += (diff*diff)
    }
    w = w.t
    h = h.t
    erroracc += err
    countacc += count
  }

  def main(args: Array[String]){
    val data_path = args(0)
    val rank = args(1).toInt
    val executor_num = args(2).toInt
    val maxiter = args(3).toInt
    var stepsize = 0.001
    var decay = 0.99

    val program_start_t = System.currentTimeMillis
    PrintSetting(data_path, rank, executor_num)
    val sparkConf = new SparkConf().setAppName("SGDMF-SCALA")
    val sc = new SparkContext(sparkConf)

    val data_raw = sc.textFile(data_path, executor_num).map(line => (line.split(",")(0).toInt,line.split(",")(1).toInt,line.split(",")(2).toFloat))
    val num_x_ids = data_raw.map(a => a._1).reduce((a,b) => if(a > b) a else b)+1
    val num_y_ids = data_raw.map(a => a._2).reduce((a,b) => if(a > b) a else b)+1
    println("[debug] x_ids : %d y_idx: %d ".format(num_x_ids, num_y_ids))  
    println("[debug] x_block dim : %d y_block dim: %d ".format(get_block_dim(num_x_ids, executor_num, 0), get_block_dim(num_y_ids, executor_num, 0)))

    val data_block = data_raw.map(a => (get_block_id(num_x_ids, executor_num, a._1), a)).partitionBy(new HashPartitioner(executor_num)).
      mapPartitionsWithIndex((in, it) => makepart(in, it, num_y_ids, executor_num), true)

    // sanity check purpose only
    //    val rates = data_block.map(a => a._2.map(a=>a.length).reduce(_+_)).reduce(_+_)
    //    val blocks = data_block.map(a => a._2.length).reduce(_+_)
    //    println("[Debug] SanityChecking workers(%d) blocks(%d), rates(%d), executor_num(%d)".format(data_block.count(), blocks, rates, executor_num))
    //    System.out.println("[USER INFO]: data(%s) rank(%d) workers(%d) rates(%d)\n".format(data_path, rank, executor_num, rates))

    // generate w/h factor matrices with random values
    val blockids = sc.parallelize(0 until executor_num, executor_num).persist()
    val w_block = blockids.map(a => generate_init(a, rank, get_block_dim(num_x_ids, executor_num, a)));
    val h_block = blockids.map(a => generate_init(a, rank, get_block_dim(num_y_ids, executor_num, a)));
    //    println("[Debug] Wblock count(%d) Hblock count(%d) \n".format(w_block.count(), h_block.count()))

    //merge R and W
    val data_block_with_w = data_block.join(w_block, numPartitions=executor_num)
    //println("[Debug] RWblock count(%d)  \n".format(data_block_with_w.count()))
    val data_array_with_w= ArrayBuffer(data_block_with_w)
    val h_array=ArrayBuffer(h_block)
    val cjoined4es=ArrayBuffer[org.apache.spark.rdd.RDD[(Int, ((Array[Array[(Int, Int, Float)]], breeze.linalg.DenseMatrix[Double]), breeze.linalg.DenseMatrix[Double]))]]()

    val loading_end_t = System.currentTimeMillis
    val first_iter_start_t = System.currentTimeMillis
    println("@@@@@ data loading latency: %f \n\n\n".format((loading_end_t - program_start_t)/1000.0))
    System.out.println("[USER INFO]: data loading latency including rate counting\n".format((loading_end_t - program_start_t)/1000.0))

    val eval_h_array=ArrayBuffer[org.apache.spark.rdd.RDD[(Int, breeze.linalg.DenseMatrix[Double])]]()

    for(iteration <- 0 until maxiter){
      for(subepoch <- 0 until executor_num){
        val idx= iteration*executor_num + subepoch
        cjoined4es += data_array_with_w(idx).join(h_array(idx), executor_num).
          mapPartitions(a => compute_a_block(a, subepoch, num_x_ids, num_y_ids, executor_num, stepsize), true).persist(StorageLevel.MEMORY_AND_DISK_SER)
        data_array_with_w += cjoined4es(idx).mapPartitions(x=> wrapper(x), true).persist(StorageLevel.MEMORY_AND_DISK_SER)
        h_array += cjoined4es(idx).map(x => update_h_ids_onlymap(x, executor_num)).persist(StorageLevel.MEMORY_AND_DISK_SER)
      }

      val error = sc.accumulator(0.0, "error")
      val counta = sc.accumulator(0, "ratecount")

      for(subepoch <- 0 until executor_num){
        val idx = iteration*executor_num + subepoch
        eval_h_array += h_array(iteration*executor_num + executor_num).map(x => update_eval_h_ids_onlymap(x, subepoch, executor_num)).persist(StorageLevel.MEMORY_AND_DISK_SER)
        data_array_with_w(iteration*executor_num +executor_num).join(eval_h_array(idx), executor_num).foreach(x => eval_a_block(x, subepoch, num_x_ids, num_y_ids, executor_num, error, counta))
      }

      val iter_end_t = System.currentTimeMillis
      println("@@@@@@@ iteration %d error %f count : %d elap_from_first_iter(%f) elap_from_start(%f) \n\n\n\n".
        format(iteration, error.value, counta.value, (iter_end_t-first_iter_start_t)/1000.0,(iter_end_t-program_start_t)/1000.0))
      System.out.println("[USER INFO] iteration %d error %f count : %d elap_from_first_iter(%f) elap_from_start(%f) \n".
        format(iteration, error.value, counta.value, (iter_end_t-first_iter_start_t)/1000.0,(iter_end_t-program_start_t)/1000.0))

      stepsize = stepsize * decay;
    }    
    println("END OF COMPUTE")
  }// end of main 
} // end of Object
