{ A unit containing TMultiArray, multidimensional array data structure with
  single precision and arbitrary number of dimension. }
unit multiarray;

{$mode objfpc}
{$H+}
{$modeSwitch advancedRecords}
{$inline on}

interface

uses
  Classes, SysUtils, math, StrUtils, DateUtils;

type
  TLongVector = array of longint;
  TLongVectorArr = array of TLongVector;
  TSingleVector = array of single;
  TGenFunc = function(Params: array of single): single;
  TUFunc = function(a: single; params: array of single): single;
  TBFunc = function(a, b: single): single;

type

  { Multidimensional array data structure. }

  TMultiArray = record
  private
    FIsContiguous: boolean;
    function GetNDims: integer;
    function GetSize: longint;
    function OffsetToStrided(Offset: longint): longint; inline;
  public
    DataOffset: longint;
    Data:  array of single;
    Indices: TLongVectorArr;
    Shape: TLongVector;
    Strides: TLongVector;
    { Returns contiguous copy of this array }
    function Contiguous: TMultiArray;
    { Perform copying of this array. By default, the actual data is referenced
      rather than copied. If Deep is true, then the actual data is copied. }
    function Copy(Deep: boolean = True): TMultiArray;
    { Get i-th item of the flattened array }
    function Get(i: longint): single; overload;
    { Returns a TSingleVector containing "strided" data, instead of the actual
      data. }
    function GetVirtualData: TSingleVector;
    { Convert Self to a single-typed scalar if NDims=0 }
    function Item: Single;
    { Multiply self with other 2-dimensional TMultiArray }
    function Matmul(Other: TMultiArray): TMultiArray;
    { Reshape a TMultiArray into a specified shape }
    function Reshape(NewShape: TLongVector): TMultiArray;
    { Perform multidimensional slicing.
      For now Slice will return contiguous. }
    function Slice(idx: array of TLongVector): TMultiArray;
    function SliceBool(BoolIdx: array of TLongVector): TMultiArray;
    function Squeeze: TMultiArray;
    { Perform transpose. If NDims > 2, the reverse the axis }
    function T: TMultiArray;
    procedure Put(i: array of integer; x: single);
    procedure ResetIndices;
    property IsContiguous: boolean read FIsContiguous write FIsContiguous;
    property NDims: integer read GetNDims;
    property Size: longint read GetSize;
    property Items[idx: array of TLongVector]: TMultiArray read Slice; default;
  end;

  TBroadcastResult = record
    A, B: TMultiArray;
  end;

  TUniqueResult = record
    UniqueValues, Counts: TMultiArray;
  end;

  TNDIter = record
    Shape: TLongVector;
    Strides: TLongVector;
    CurrentVirt: TLongVector;
    Current: TLongVector;
    CurrentOffset: longint;
    NDims: longint;
    function Next(X: TMultiArray): single;
    procedure Advance(X: TMultiArray);
    procedure Reset(X: TMultiArray);
  end;

  TBFuncWorker = class(TThread)
    itA, itB: TNDIter;
    fA, fB, fC: TMultiArray;
    fBFunc: TBFunc;
    fStartingOffset, fCount: longint;
    constructor Create(A, B: TMultiArray; var C: TMultiArray; BFunc: TBFunc;
    StartingOffset, count: longint; CreateSuspended: Boolean; const StackSize: SizeUInt=
      DefaultStackSize);
    procedure Execute(); override;
  end;

  function _ALL_: TLongVector;
  function AllocateMultiArray(Size: longint; AllocateData: boolean=true): TMultiArray;
  function Any(X: TMultiArray; Axis: longint = -1): TMultiArray;
  function ApplyBFunc(A, B: TMultiArray; BFunc: TBFunc; PrintDebug: Boolean = False;
    FuncName: string = ''): TMultiArray;
  function ApplyBFuncThreaded(A, B: TMultiArray; BFunc: TBFunc; PrintDebug: Boolean = False;
    FuncName: string = ''): TMultiArray;
  function ApplyUFunc(A: TMultiArray; UFunc: TUFunc; Params: array of single): TMultiArray;
  function ArrayEqual(A, B: TMultiArray; Tol: single=0): boolean;
  function ArraySort(A: TMultiArray; Axis: integer = -1): TMultiArray;
  function AsStrided(A: TMultiArray; Shape, Strides: TLongVector): TMultiArray;
  function BroadcastArrays(A, B: TMultiArray): TBroadcastResult;
  function Contiguous(X: TMultiArray): TMultiArray;
  function CreateEmptyFTensor(Contiguous: boolean = True): TMultiArray;
  function CreateMultiArray(AData: array of single): TMultiArray;
  function CreateMultiArray(AData: single): TMultiArray;
  function DynArrayToVector(A: array of longint): TLongVector;
  function GenerateMultiArray(Shape: array of longint; GenFunc: TGenFunc;
    Params: array of single): TMultiArray;
  function GetVirtualData(A: TMultiArray): TSingleVector;
  function Im2Col(X: TMultiArray; FH, FW, Stride, Pad: longint): TMultiArray;
  { Get linear index of x in A. Returns -1 if not found. }
  function IndexOf(x: Single; A: TMultiArray): longint;
  function IndexToStridedOffset(Index: array of longint; Strides: TLongVector): longint;
  function MatToStringList(A: TMultiArray; delimiter: string=','): TStringList;
  function OffsetToIndex(A: TMultiArray; Offset: longint): TLongVector;
  function Range(Start, Stop, step: longint): TLongVector; overload;
  function Range(Start, Stop: longint): TLongVector; overload;
  function Ravel(A: TMultiArray): TMultiArray;
  function ReadCSV(FileName: string): TMultiArray;
  function ShapeToStrides(AShape: TLongVector): TLongVector;
  function Transpose(A: TMultiArray): TMultiArray;
  function Transpose(A: TMultiArray; NewAxis: TLongVector): TMultiArray;
  function Unique(A: TMultiArray; Axis: integer = -1; ReturnCounts: boolean = False): TUniqueResult;
  function VStack(arr: array of TMultiArray): TMultiArray;

  procedure DebugMultiArray(A: TMultiArray);
  procedure EnsureNDims(A: TMultiArray; NDims: integer);
  procedure PrintMultiArray(A: TMultiArray; DecPlace: longint = 2);
  { Shuffle an array based on the first index }
  procedure Permute(var A: TMultiArray);
  procedure SqueezeMultiArrayAt(var A: TMultiArray; axis: integer);
  procedure SqueezeMultiArray(var A: TMultiArray);
  procedure WriteCSV(A: TMultiArray; filename: string);

  function CopyVector(V: TSingleVector): TSingleVector;
  function CopyVector(V: TLongVector): TLongVector;
  procedure ShuffleVector(var V: array of longint);
  procedure ShuffleVector(var V: array of single);
  function VectorEqual(A, B: TSingleVector): boolean;
  function VectorEqual(A, B: TLongVector): boolean;

  function Add(A, B: TMultiArray): TMultiArray;
  function Matmul(A, B: TMultiArray): TMultiArray;
  function Maximum(A, B: TMultiArray): TMultiArray;
  function Minimum(A, B: TMultiArray): TMultiArray;
  function Multiply(A, B: TMultiArray): TMultiArray;
  function Power(A, B: TMultiArray): TMultiArray; overload;

  { @exclude } operator in (A: single; B: TSingleVector) C: boolean;

  { @exclude } operator + (A, B: TMultiArray) C: TMultiArray;
  { @exclude } operator - (A: TMultiArray) B: TMultiArray;
  { @exclude } operator - (A, B: TMultiArray) C: TMultiArray;
  { @exclude } operator * (A, B: TMultiArray) C: TMultiArray;
  { @exclude } operator / (A, B: TMultiArray) C: TMultiArray;
  { @exclude } operator ** (A, B: TMultiArray) C: TMultiArray;
  { @exclude } operator = (A, B: TMultiArray) C: TMultiArray;
  { @exclude } operator > (A, B: TMultiArray) C: TMultiArray;
  { @exclude } operator >= (A, B: TMultiArray) C: TMultiArray;
  { @exclude } operator < (A, B: TMultiArray) C: TMultiArray;
  { @exclude } operator <= (A, B: TMultiArray) C: TMultiArray;
  { @exclude } operator in (A: TMultiArray; B: array of TMultiArray) C: boolean;

  { @exclude } operator := (A: single) B: TMultiArray;
  { @exclude } operator := (A: array of single) B: TMultiArray;
  { @exclude } operator explicit(A: single) B: TMultiArray;
  { @exclude } operator :=(A: array of longint) B: TLongVector;
  { @exclude } operator explicit(A: array of longint) B: TLongVector;
  { @exclude } operator :=(A: TLongVector) B: TSingleVector;
  { @exclude } operator explicit(A: TLongVector) B: TSingleVector;
  { @exclude } operator :=(A: TSingleVector) B: TLongVector;
  { @exclude } operator explicit(A: TSingleVector) B: TLongVector;
  { @exclude } operator :=(A: longint) B: TLongVector;
  { @exclude } operator explicit(A: longint) B: TLongVector;

  // 1-dim array to TSingleVector
  { @exclude } operator :=(A: TMultiArray) B: TSingleVector;
  { @exclude } operator explicit(A: TMultiArray) B: TSingleVector;
  // 1-dim array to TLongVector
  { @exclude } operator :=(A: TMultiArray) B: TLongVector;
  { @exclude } operator explicit(A: TMultiArray) B: TLongVector;

const
  MAX_THREAD_NUM = 4;

var
  GLOBAL_FUNC_DEBUG: boolean;

implementation

uses
  numerik, numerik.blas;

  generic function _CopyVector<T>(v: T): T;
  var
    i: longint;
  begin
    Result := nil;
    SetLength(Result, Length(v));
    for i := 0 to High(v) do
      Result[i] := v[i];
  end;

  generic function Prod<T>(Data: array of T): T;
  var
    x: T;
  begin
    Result := 1;
    for x in Data do
      Result := Result * x;
  end;

  generic function Reverse<T>(Data: T): T;
  var
    i: longint;
  begin
    Result := nil;
    SetLength(Result, Length(Data));
    For i := High(Data) downto 0 do
      Result[High(Data) - i] := Data[i];
  end;

  generic function _VectorEqual<T>(A, B: T): boolean;
  var
    i: longint;
  begin
    if (Length(A) <> Length(B)) then Exit(false);
    Result := True;
    for i := 0 to High(A) do
      Result := Result and (A[i] = B[i]);
  end;

  generic procedure DeleteFromVector<T>(var Data: T; At, Count: longint);
  var
    i, Offset: longint;
  begin
    Offset := At + Count;
    for i := Offset to High(Data) do
      Data[i - Count] := Data[i];
    SetLength(Data, Length(Data) - Count);
  end;

  generic procedure PrintVector<T>(Data: T);
  var
    i: longint;
  begin
    for i := 0 to Length(Data) - 1 do
    begin
      Write(Data[i] : 2);
      if i < Length(Data) - 1 then
        Write(', ');
    end;
    WriteLn;
  end;

  function _ALL_: TLongVector;
  begin
    Exit([]);
  end;

  function AllocateMultiArray(Size: longint; AllocateData: boolean=true): TMultiArray;
  begin
    Result.Data := nil;
    if AllocateData then
      SetLength(Result.Data, Size);
    SetLength(Result.Shape, 1);
    Result.IsContiguous := True;
    Result.Shape[0] := Size;
    Result.Strides := ShapeToStrides(Result.Shape);
    Result.ResetIndices;
    Result.DataOffset := 0;
  end;

  function BroadcastArrays(A, B: TMultiArray): TBroadcastResult;
  var
    AdjShapeA, AdjShapeB, AdjStridesA, AdjStridesB: TLongVector;
    i, NewNDims: integer;
  begin
    NewNDims := Max(A.NDims, B.NDims);
    AdjShapeA := specialize Reverse<TLongVector>(A.Shape);
    AdjShapeB := specialize Reverse<TLongVector>(B.Shape);
    AdjStridesA := specialize Reverse<TLongVector>(A.Strides);
    AdjStridesB := specialize Reverse<TLongVector>(B.Strides);

    { Broadcast dimension is always max(A.NDims, B.NDims) }
    if A.NDims < NewNDims then
    begin
      SetLength(AdjShapeA, NewNDims);
      SetLength(AdjStridesA, NewNDims)
    end
    else
    begin
      SetLength(AdjShapeB, NewNDims);
      SetLength(AdjStridesB, NewNDims)
    end;

    for i := 0 to NewNDims - 1 do
      if (AdjShapeA[i] <> AdjShapeB[i]) then
      begin
        if (AdjShapeA[i] < AdjShapeB[i]) and (AdjShapeA[i] <= 1) then
        begin
          AdjStridesA[i] := 0;
          AdjShapeA[i] := AdjShapeB[i]
        end
        else if (AdjShapeB[i] < AdjShapeA[i]) and (AdjShapeB[i] <= 1) then
        begin
          AdjStridesB[i] := 0;
          AdjShapeB[i] := AdjShapeA[i]
        end
        else
          raise Exception.Create('Broadcasting failed.');
      end;

    Result.A := AsStrided(A, specialize Reverse<TLongVector>(AdjShapeA),
                             specialize Reverse<TLongVector>(AdjStridesA));
    Result.A.ResetIndices;
    Result.B := AsStrided(B, specialize Reverse<TLongVector>(AdjShapeB),
                             specialize Reverse<TLongVector>(AdjStridesB));
    Result.B.ResetIndices;
  end;

  function Contiguous(X: TMultiArray): TMultiArray;
  var
    it: TNDIter;
    i: longint;
  begin
    it.Reset(X);
    SetLength(Result.Data, X.Size);
    for i := 0 to X.Size - 1 do
    begin
      it.Advance(X);
      Result.Data[i] := X.Data[it.CurrentOffset];
    end;
    Result.Shape := CopyVector(X.Shape);
    Result.Strides := ShapeToStrides(X.Shape);
    Result.ResetIndices;
    Result.IsContiguous:=True;
  end;

  function CreateEmptyFTensor(Contiguous: boolean = True):TMultiArray;
  begin
    Result.IsContiguous := Contiguous;
    Result.DataOffset := 0;
  end;

  function CreateMultiArray(AData: array of single): TMultiArray;
  var
    i: longint;
  begin
    Result := CreateEmptyFTensor(True);
    SetLength(Result.Data, Length(AData));
    for i := 0 to Length(AData) - 1 do
      Result.Data[i] := AData[i];
    SetLength(Result.Shape, 1);
    Result.Shape[0] := Length(AData);
    Result.Strides := ShapeToStrides(Result.Shape);
    Result.ResetIndices;
  end;

  function CreateMultiArray(AData: single): TMultiArray;
  begin
    Result := CreateMultiArray([AData]);
    SqueezeMultiArray(Result);
  end;

  function ArrayEqual(A, B: TMultiArray; Tol: single=0): boolean;
  var
    InTol: boolean=True;
  begin
    if not VectorEqual(A.Shape, B.Shape) then Exit(False);
    InTol := Mean(Abs(A - B)).Item <= Tol;
    Exit(InTol);
  end;

  generic procedure Quicksort<T>(var arr: array of T);
    procedure QSort(Left, Right: longint);
    var
      i, j: longint;
      tmp, pivot: T;
    begin
      i := Left;
      j := Right;
      pivot := arr[(Left + Right) shr 1];
      repeat
        while pivot > arr[i] do
          Inc(i);
        while pivot < arr[j] do
          Dec(j);
        if i <= j then
        begin
          tmp := arr[i];
          arr[i] := arr[j];
          arr[j] := tmp;
          Dec(j);
          Inc(i);
        end;
      until i > j;
      if Left < j then
        QSort(Left, j);
      if i < Right then
        QSort(i, Right);
    end;
  begin
    QSort(0, Length(arr) - 1);
  end;

  function ArraySort(A: TMultiArray; Axis: integer): TMultiArray;
  var
    arr: TSingleVector;
  begin
    if Axis > -1 then
      raise Exception.Create('Sorting with axis > -1 is not supported yet.');

    arr := CopyVector(A.GetVirtualData);
    specialize Quicksort<single>(arr);
    Exit(arr);
  end;

  function AsStrided(A: TMultiArray; Shape, Strides: TLongVector): TMultiArray;
  begin
    Result := CreateEmptyFTensor(false);
    Result.Data := A.Data;
    Result.Shape := Shape;
    Result.Strides := Strides;
    Result.ResetIndices;
  end;

  function Any(X: TMultiArray; Axis: longint = -1): TMultiArray;
  var
    idx: TLongVectorArr;
    vals: TSingleVector;
    i: longint;
  begin
    if Axis + 1 > X.NDims then
      raise Exception.Create('Axis is greater than X.NDims.');

    SetLength(idx, X.NDims);
    for i := 0 to High(idx) do
      idx[i] := [];

    if Axis = -1 then
      Exit(Sum(X) > 0);

    SetLength(vals, X.Shape[Axis]);
    for i := 0 to X.Shape[Axis] - 1 do
    begin
      idx[Axis] := i;
      Vals[i] := Any(X[idx]).Item;
    end;
    Exit(Vals);
  end;


  function ApplyBFunc(A, B: TMultiArray; BFunc: TBFunc; PrintDebug: Boolean = False;
    FuncName: string = ''): TMultiArray;
  var
    i, Offset: longint;
    x: single;
    BcastResult: TBroadcastResult;
    TimeThen: TDateTime;
    ItA, ItB: TNDIter;
    OutArr: TMultiArray;
  begin

    if (A.NDims = 0) and (B.NDims = 0) then
      Exit(BFunc(A.Data[0], B.Data[0]))
    else if (B.NDims = 0) then
    begin
      Result := AllocateMultiArray(A.Size).Reshape(A.Shape);
      x := B.Data[0];
      for i := 0 to High(A.Data) do
        Result.Data[i] := BFunc(A.Data[i], x);
    end
    else if (A.NDims = 0) then
    begin
      Result := AllocateMultiArray(B.Size).Reshape(B.Shape);
      x := A.Data[0];
      for i := 0 to High(B.Data) do
        Result.Data[i] := BFunc(x, B.Data[i]);
    end;

    if not(VectorEqual(A.Shape, B.Shape)) then
    begin
      BcastResult := BroadcastArrays(A, B);

      Result := ApplyBFunc(BcastResult.A.Contiguous, BcastResult.B.Contiguous, BFunc, PrintDebug, FuncName);
      Result.IsContiguous:=True;
    end else
    begin

      Result := AllocateMultiArray(A.Size);
      Result := Result.Reshape(A.Shape); // should be reset strides
      Result.IsContiguous := True;

      if (PrintDebug or GLOBAL_FUNC_DEBUG) then TimeThen := Now;

      if not(A.IsContiguous) and not(B.IsContiguous) then
      begin
        ItA.Reset(A);
        ItB.Reset(B);
        for i := 0 to A.Size div 8 - 1 do
        begin
          Result.Data[i * 8 + 0] := BFunc(ItA.Next(A), ItB.Next(B));
          Result.Data[i * 8 + 1] := BFunc(ItA.Next(A), ItB.Next(B));
          Result.Data[i * 8 + 2] := BFunc(ItA.Next(A), ItB.Next(B));
          Result.Data[i * 8 + 3] := BFunc(ItA.Next(A), ItB.Next(B));
          Result.Data[i * 8 + 4] := BFunc(ItA.Next(A), ItB.Next(B));
          Result.Data[i * 8 + 5] := BFunc(ItA.Next(A), ItB.Next(B));
          Result.Data[i * 8 + 6] := BFunc(ItA.Next(A), ItB.Next(B));
          Result.Data[i * 8 + 7] := BFunc(ItA.Next(A), ItB.Next(B));
        end;
        for i := (A.Size div 8) * 8 to (A.Size div 8) * 8 + (A.Size mod 8) - 1 do
          Result.Data[i] := BFunc(ItA.Next(A), ItB.Next(B));
      end
      else if (A.IsContiguous) and not(B.IsContiguous) then
      begin
        ItB.Reset(B);
        for i := 0 to A.Size div 8 - 1 do
        begin
          Result.Data[i * 8 + 0] := BFunc(A.Data[i * 8 + 0], ItB.Next(B));
          Result.Data[i * 8 + 1] := BFunc(A.Data[i * 8 + 1], ItB.Next(B));
          Result.Data[i * 8 + 2] := BFunc(A.Data[i * 8 + 2], ItB.Next(B));
          Result.Data[i * 8 + 3] := BFunc(A.Data[i * 8 + 3], ItB.Next(B));
          Result.Data[i * 8 + 4] := BFunc(A.Data[i * 8 + 4], ItB.Next(B));
          Result.Data[i * 8 + 5] := BFunc(A.Data[i * 8 + 5], ItB.Next(B));
          Result.Data[i * 8 + 6] := BFunc(A.Data[i * 8 + 6], ItB.Next(B));
          Result.Data[i * 8 + 7] := BFunc(A.Data[i * 8 + 7], ItB.Next(B));
        end;
        for i := (A.Size div 8) * 8 to (A.Size div 8) * 8 + (A.Size mod 8) - 1 do
          Result.Data[i] := BFunc(A.Data[i], ItB.Next(B));
      end
      else if not(A.IsContiguous) and (B.IsContiguous) then
      begin
        ItA.Reset(A);
        for i := 0 to A.Size div 8 - 1 do
        begin
          Result.Data[i * 8 + 0] := BFunc(ItA.Next(A), B.Data[i * 8 + 0]);
          Result.Data[i * 8 + 1] := BFunc(ItA.Next(A), B.Data[i * 8 + 1]);
          Result.Data[i * 8 + 2] := BFunc(ItA.Next(A), B.Data[i * 8 + 2]);
          Result.Data[i * 8 + 3] := BFunc(ItA.Next(A), B.Data[i * 8 + 3]);
          Result.Data[i * 8 + 4] := BFunc(ItA.Next(A), B.Data[i * 8 + 4]);
          Result.Data[i * 8 + 5] := BFunc(ItA.Next(A), B.Data[i * 8 + 5]);
          Result.Data[i * 8 + 6] := BFunc(ItA.Next(A), B.Data[i * 8 + 6]);
          Result.Data[i * 8 + 7] := BFunc(ItA.Next(A), B.Data[i * 8 + 7]);
        end;
        for i := (A.Size div 8) * 8 to (A.Size div 8) * 8 + (A.Size mod 8) - 1 do
          Result.Data[i] := BFunc(ItA.Next(A), B.Data[i]);
      end
      else
      begin
        for i := 0 to A.Size div 8 - 1 do
        begin
          Offset := i * 8;
          Result.Data[Offset + 0] := BFunc(A.Data[Offset + 0], B.Data[Offset + 0]);
          Result.Data[Offset + 1] := BFunc(A.Data[Offset + 1], B.Data[Offset + 1]);
          Result.Data[Offset + 2] := BFunc(A.Data[Offset + 2], B.Data[Offset + 2]);
          Result.Data[Offset + 3] := BFunc(A.Data[Offset + 3], B.Data[Offset + 3]);
          Result.Data[Offset + 4] := BFunc(A.Data[Offset + 4], B.Data[Offset + 4]);
          Result.Data[Offset + 5] := BFunc(A.Data[Offset + 5], B.Data[Offset + 5]);
          Result.Data[Offset + 6] := BFunc(A.Data[Offset + 6], B.Data[Offset + 6]);
          Result.Data[Offset + 7] := BFunc(A.Data[Offset + 7], B.Data[Offset + 7]);
        end;
        for i := (A.Size div 8) * 8 to (A.Size div 8) * 8 + (A.Size mod 8) - 1 do
          Result.Data[i] := BFunc(A.Data[i], B.Data[i]);
      end;

      if (PrintDebug or GLOBAL_FUNC_DEBUG) then
        WriteLn('Function ' + FuncName + ' executed in ',
                MilliSecondsBetween(TimeThen, Now), ' ms');
    end;
  end;

  function ApplyBFuncThreaded(A, B: TMultiArray; BFunc: TBFunc; PrintDebug: Boolean = False;
    FuncName: string = ''): TMultiArray;
  var
    i, NumLeft, NumWorkers: longint;
    BcastResult: TBroadcastResult;
    TimeThen: TDateTime;
    ItA, ItB: TNDIter;

    worker1: TBFuncWorker;
    workers: array of TBFuncWorker;
    Strides: TLongVector;
  begin
    if (PrintDebug or GLOBAL_FUNC_DEBUG) then TimeThen := Now;

    if (A.NDims = 0) and (B.NDims = 0) then
      Exit(BFunc(A.Data[0], B.Data[0]));

    if not(VectorEqual(A.Shape, B.Shape)) then
    begin
      BcastResult := BroadcastArrays(A, B);

      Result := ApplyBFuncThreaded(BcastResult.A.Contiguous, BcastResult.B.Contiguous,
        BFunc, PrintDebug, FuncName);
      Result.IsContiguous:=True;
    end else
    begin
      Result := AllocateMultiArray(A.Size);
      Result := Result.Reshape(A.Shape); // should be reset strides
      Result.IsContiguous := True;

      NumWorkers := MAX_THREAD_NUM;
      SetLength(workers, NumWorkers);
      for i := 0 to NumWorkers - 1 do
      begin
        workers[i] := TBFuncWorker.Create(A, B, Result, BFunc, i * (Result.Size div NumWorkers),
          Result.Size div NumWorkers, True);
        workers[i].FreeOnTerminate := False;

        workers[i].itA.Reset(A);
        workers[i].itB.Reset(B);

        Strides := ShapeToStrides(Result.Shape);
        workers[i].itA.CurrentVirt := OffsetToIndex(Result, i * (Result.Size div NumWorkers));
        workers[i].itB.CurrentVirt := OffsetToIndex(Result, i * (Result.Size div NumWorkers));

        if i = NumWorkers - 1 then
        begin
          workers[i].fCount := workers[i].fCount + (A.Size mod NumWorkers);
        end;
      end;

      for i := 0 to NumWorkers - 1 do
        workers[i].Start();

      for i := 0 to NumWorkers - 1 do
        workers[i].WaitFor();

      for i := 0 to NumWorkers - 1 do
        workers[i].Free;

      if (PrintDebug or GLOBAL_FUNC_DEBUG) then
        WriteLn('Function ' + FuncName + ' executed in ',
                MilliSecondsBetween(TimeThen, Now), ' ms');
    end;
  end;

  function ApplyUFunc(A: TMultiArray; UFunc: TUFunc; Params: array of single): TMultiArray;
  var
    i: longint;
  begin
    Result := A.Copy();

    for i := 0 to A.Size div 8 - 1 do
        begin
          Result.Data[i * 8 + 0] := UFunc(A.Data[i * 8 + 0], Params);
          Result.Data[i * 8 + 1] := UFunc(A.Data[i * 8 + 1], Params);
          Result.Data[i * 8 + 2] := UFunc(A.Data[i * 8 + 2], Params);
          Result.Data[i * 8 + 3] := UFunc(A.Data[i * 8 + 3], Params);
          Result.Data[i * 8 + 4] := UFunc(A.Data[i * 8 + 4], Params);
          Result.Data[i * 8 + 5] := UFunc(A.Data[i * 8 + 5], Params);
          Result.Data[i * 8 + 6] := UFunc(A.Data[i * 8 + 6], Params);
          Result.Data[i * 8 + 7] := UFunc(A.Data[i * 8 + 7], Params);
        end;
        for i := (A.Size div 8) * 8 to (A.Size div 8) * 8 + (A.Size mod 8) - 1 do
          Result.Data[i] := UFunc(A.Data[i], Params);

    //Result.ResetIndices;
  end;

  function DynArrayToVector(A: array of longint): TLongVector;
  var
    i: longint;
  begin
    Result := nil;
    SetLength(Result, Length(A));
    for i := 0 to High(A) do
      Result[i] := A[i];
  end;

  procedure DebugMultiArray(A: TMultiArray);
  var
    i: longint;
  begin
    writeln('Data offset    :', A.DataOffset);
    writeln('Is contiguous  :', A.IsContiguous);
    writeln('NDims          :', A.NDims);
    Write  ('Shape          :'); specialize PrintVector<TLongVector>(A.Shape);
    writeln('Size (actual)  :', Length(A.Data));
    writeln('Size (virtual) :', A.Size);
    Write  ('Strides        :'); specialize PrintVector<TLongVector>(A.Strides);
    WriteLn('----------------');
    WriteLn('Indices        :');
    WriteLn('----------------');
    for i := 0 to Length(A.Indices) - 1 do
    begin
      Write('> Axis ', i, ' ==> '); specialize PrintVector<TLongVector>(A.Indices[i]);
    end;
  end;

  procedure EnsureNDims(A: TMultiArray; NDims: integer);
  begin
    if A.NDims <> NDims then
      raise Exception.Create('A.NDims must be equals to ' + IntToStr(NDims) + '. ' +
                              'Got ' + IntToStr(A.NDims) + ' instead.');
  end;

  procedure PrintMatrix(A: TMultiArray);
  var
    r, c, cnt: longint;
  begin
    if A.NDims > 2 then
      raise Exception.Create('Cannot print array with NDims > 2');
    cnt := 0;
    for r := 0 to A.Shape[0] - 1 do
    begin
      for c := 0 to A.Shape[1] - 1 do
      begin
        if c = 0 then Write('[');
        Write(A.Get(cnt) : 5 : 2);
        if c < A.Shape[1] - 1 then Write(', ');
        if c = A.Shape[1] - 1 then Write(']');
        Inc(cnt);
      end;

      if r < (A.Shape[0] - 1) then
        WriteLn;
    end;
  end;

  procedure PrintMultiArray(A: TMultiArray; DecPlace: longint = 2);
  var
    Digit, MaxDims: integer;
    MaxNUm: single;
    s: string;


    procedure PrintHelper(A: TMultiArray; it: longint);
    var
      i, j: integer;
      tmp: TMultiArray;
    begin
      if A.NDims = 0 then
      begin
        for j := 0 to A.Size - 1 do
        begin
          s := s + Format('%' + IntToStr(Digit + DecPlace + 2) + '.' +
                          IntToStr(DecPlace) +'f, ', [A.Data[j]]);
        end;
        Exit;
      end;

      if it > 0 then
      begin
        s := s + ',' + DupeString(sLineBreak, A.NDims);
        s := s + DupeString(' ', MaxDims - A.NDims);
      end;
      s := s + '[';
      for i := 0 to A.Shape[0] - 1 do
      begin
        tmp := A.Slice([[i]]);
        SqueezeMultiArrayAt(tmp, 0);
        PrintHelper(tmp, i);
      end;
      s := s + ']';
    end;

  begin
    s := '';
    MaxDims := A.NDims;
    MaxNum := MaxValue(Abs(A).Data);
    Digit := Ceil(Log10(MaxNum + 1));
    PrintHelper(A, 0);
    WriteLn(s);
  end;

  procedure WriteCSV(A: TMultiArray; filename: string);
  var
    sl: TStringList;
  begin
    sl := MatToStringList(A, ',');
    sl.SaveToFile(filename);
    sl.Free;
  end;

  function CopyVector(v: TSingleVector): TSingleVector;
  begin
    Exit(specialize _CopyVector<TSingleVector>(v));
  end;

  function CopyVector(v: TLongVector): TLongVector;
  begin
    Exit(specialize _CopyVector<TLongVector>(v));
  end;

  generic procedure _ShuffleVector<T>(var V: array of T);
  var
    i, p: integer;
    tmp: T;

  begin
    { Fisher-Yates shuffle algorithm }
    for i := Length(V) downto 1 do
    begin
      p := RandomRange(0, i);
      tmp := V[i - 1];
      V[i - 1] := V[p];
      V[p] := tmp;
    end;
  end;

  procedure ShuffleVector(var V: array of longint);
  begin
    specialize _ShuffleVector<longint>(V);
  end;

  procedure ShuffleVector(var V: array of single);
  begin
    specialize _ShuffleVector<single>(V);
  end;

  function VectorEqual(A, B: TSingleVector): boolean;
  begin
    Exit(specialize _VectorEqual<TSingleVector>(A, B));
  end;

  function VectorEqual(A, B: TLongVector): boolean;
  begin
    Exit(specialize _VectorEqual<TLongVector>(A, B));
  end;

  function GenerateMultiArray(Shape: array of longint; GenFunc: TGenFunc;
    Params: array of single): TMultiArray;
  var
    i: longint;
  begin
    Result := AllocateMultiArray(specialize Prod<longint>(Shape));
    for i := 0 to High(Result.Data) do
      Result.Data[i] := GenFunc(Params);
    Result := Result.Reshape(Shape);
  end;

  function GetVirtualData(A: TMultiArray): TSingleVector;
  var
    i: longint;
  begin
    Result := nil;
    SetLength(Result, A.Size);
    for i := 0 to A.Size - 1 do
      Result[i] := A.Get(i);
  end;

  function MatToStringList(A: TMultiArray; delimiter: string=','): TStringList;
  var
    sl: TStringList;
    row, col: longint;
    s: string;
  begin
    if A.NDims > 2 then
      raise Exception.Create('Cannot save array with NDims>2 to CSV.');

    sl := TStringList.Create;
    for row := 0 to A.Shape[0] - 1 do
    begin
      s := '';
      for col := 0 to A.Shape[1] - 1 do
      begin
        s := s + FloatToStr(A[[row, col]].Item);
        if col < A.Shape[1] - 1 then
          s := s + delimiter;
      end;
      sl.Add(s);
    end;
    Exit(sl);
  end;

  function OffsetToIndex(A: TMultiArray; Offset: longint): TLongVector;
  var
    cnt, i: longint;
  begin
    SetLength(Result, A.NDims);
    cnt := A.NDims - 1;
    for i := High(A.Shape) downto 0 do
    begin
      Result[cnt] := A.Indices[cnt][Offset mod A.Shape[i]];
      //Result[cnt] := Offset mod A.Shape[i];
      Offset := Offset div A.Shape[i];
      Dec(cnt);
    end;
  end;

  function Range(start, stop, step: longint): TLongVector;
  var
    idx, num, Size: longint;
  begin
    Size := Ceil(Float(stop - start) / step);
    SetLength(Result, Size);

    num := start;
    idx := 0;
    while num < stop do
    begin
      Result[idx] := num;
      Inc(idx);
      Inc(num, step);
    end;
  end;

  function Range(Start, Stop: longint): TLongVector;
  begin
    Exit(Range(Start, Stop, 1));
  end;

  function Ravel(A: TMultiArray): TMultiArray;
  begin
    Exit(A.GetVirtualData);
  end;

  function ReadCSV(FileName: string): TMultiArray;
  var
    sl: TStringList;
    Offset, RowCnt, ColCnt: integer;
    c, s: string;
    Data: TSingleVector;
  begin
    sl := TStringList.Create;
    sl.LoadFromFile(FileName);

    RowCnt := sl.Count;
    if RowCnt > 0 then
      ColCnt := Length(sl[0].Split(',')); // Assuming num of cols is consistent

    SetLength(Data, RowCnt * ColCnt);
    Offset := 0;
    for s in sl do
    begin
      for c in s.Split(',') do
      begin
        Data[Offset] := StrToFloat(c);
        Inc(Offset);
      end;
    end;
    sl.Free;

    Result := AllocateMultiArray(RowCnt * ColCnt).Reshape([RowCnt, ColCnt]);
    Result.Data := Data;
  end;

  function ShapeToStrides(AShape: TLongVector): TLongVector;
  var
    k, j, prod: integer;
  begin
    Result := nil;

    SetLength(Result, Length(AShape));
    for k := 0 to Length(Result) - 1 do
    begin
      prod := 1;
      for j := k + 1 to Length(Result) - 1 do
        prod := prod * AShape[j];
      Result[k] := prod;
    end;
  end;

  function Transpose(A: TMultiArray): TMultiArray;
  var
    i: integer;
    NewIndices: TLongVectorArr = nil;
  begin
    Result := A.Copy();
    Result.FIsContiguous := False;
    Result.Shape := specialize Reverse<TLongVector>(A.Shape);
    Result.Strides := specialize Reverse<TLongVector>(A.Strides);
    SetLength(NewIndices, Length(Result.Indices));
    for i := 0 to High(Result.Indices) do
      NewIndices[i] := Result.Indices[High(Result.Indices) - i];
    Result.Indices := NewIndices;
  end;

  function Transpose(A: TMultiArray; NewAxis: TLongVector): TMultiArray;
  var
    OutShape: TLongVector = nil;
    OutStrides: TLongVector = nil;
    i: integer;
  begin
    SetLength(OutShape, Length(NewAxis));
    SetLength(OutStrides, Length(NewAxis));
    for i := 0 to High(NewAxis) do
    begin
      OutShape[i] := A.Shape[NewAxis[i]];
      OutStrides[i] := A.Strides[NewAxis[i]];
    end;
    Exit(AsStrided(A, OutShape, OutStrides));
  end;

  function Unique(A: TMultiArray; Axis: integer = -1; ReturnCounts: boolean = False): TUniqueResult;
  var
    tmpSingles, tmpCounts: TSingleVector;
    i, NumUnique: longint;
  begin
    if Axis > 0 then
      raise Exception.Create('`Unique` along axis is not implemented yet.');

    NumUnique := 0;
    SetLength(tmpSingles, 0);
    if ReturnCounts then SetLength(tmpCounts, 0);
    for i := 0 to A.Size - 1 do
    begin
      if not (A.Get(i) in tmpSingles) then
      begin
        SetLength(tmpSingles, NumUnique + 1);
        tmpSingles[NumUnique] := A.Get(i);
        if ReturnCounts then
        begin
          SetLength(tmpCounts, NumUnique + 1);
          tmpCounts[NumUnique] := 1;
        end;
        Inc(NumUnique);
      end
      else if ReturnCounts then
        tmpCounts[IndexOf(A.Get(i), tmpSingles)] := tmpCounts[IndexOf(A.Get(i), tmpSingles)] + 1;
    end;
    Result.UniqueValues := tmpSingles;

    if ReturnCounts then Result.Counts := tmpCounts;
    //Exit(tmpSingles);
  end;

  function VStack(arr: array of TMultiArray): TMultiArray;
  var
    i, offset, Height, Width, Prod: LongInt;
    A: TMultiArray;
  begin
    if Length(arr) <= 1 then
      raise Exception.Create('VStack requires one or more TMultiArray(s).');

    { Precompute the output height and width }
    Prod := 1;
    Height := 0;
    for A in arr do
    begin
      if A.NDims > 2 then
        raise Exception.Create('VStack for NDims>2 has not been implemented.');
      Height := Height + A.Shape[0];
      Prod := Prod * A.Size;
    end;

    Width := arr[0].Shape[1];
    Result := AllocateMultiArray(Height * Width).Reshape([Height, Width]);

    offset := 0;
    for A in arr do
    begin
      for i := offset to offset + A.Size - 1 do
        Result.Data[i] := A.Get(i - offset);
      Inc(offset, A.Size);
    end;

  end;

  procedure Permute(var A: TMultiArray);
  begin
    if A.NDims = 0 then Exit;

    ShuffleVector(A.Indices[0]);
    A.IsContiguous := False;
    A := A.Contiguous;
  end;

  procedure SqueezeMultiArrayAt(var A: TMultiArray; axis: integer);
  begin
    if Length(A.Shape) > 0 then
    if A.Shape[axis] = 1 then
    begin
      specialize DeleteFromVector<TLongVector>(A.Shape, axis, 1);
      specialize DeleteFromVector<TLongVector>(A.Strides, axis, 1);
      specialize DeleteFromVector<TLongVectorArr>(A.Indices, axis, 1);
    end;
  end;

  procedure SqueezeMultiArray(var A: TMultiArray);
  var
    len, i: integer;
  begin
    len := A.NDims;
    for i := len - 1 downto 0 do
      if A.Shape[i] = 1 then
      begin
        SqueezeMultiArrayAt(A, i);
      end;
  end;

  { TBFuncWorker }

  constructor TBFuncWorker.Create(A, B: TMultiArray; var C: TMultiArray;
    BFunc: TBFunc; StartingOffset, count: longint; CreateSuspended: Boolean;
    const StackSize: SizeUInt);
  begin
    inherited Create(CreateSuspended, StackSize);
    fA := A;
    fB := B;
    fC := C;
    fBFunc := BFunc;
    fStartingOffset := StartingOffset;
    fCount := Count;
  end;

  procedure TBFuncWorker.Execute();
  var
    i: longint;
  begin
    if (fA.IsContiguous) and (fB.IsContiguous) then
    begin
      for i := fStartingOffset to fStartingOffset + fCount - 1 do
      begin
        fC.Data[i] := fBFunc(fA.Data[i], fB.Data[i]);
      end;
    end
    else
    begin
      for i := fStartingOffset to fStartingOffset + fCount - 1 do
      begin
        fC.Data[i] := fBFunc(itA.Next(fA), itB.Next(fB));
      end;
    end;
  end;

  generic function TMultiArray.OffsetToStrided(Offset: longint): longint;
  var
    r, c: longint;
    index: TLongVector;
  begin
    if IsContiguous then Exit(Offset);

    { A scalar }
    if Length(Data) = 1 then Exit(0);

    { A vector }
    if NDims = 1 then Exit(Indices[0][Offset] + DataOffset);

    { Higher rank }
    if NDims = 2 then
    begin
      r := Indices[0][Offset div Shape[1]];
      c := Indices[1][Offset mod Shape[1]];
      Exit(IndexToStridedOffset([r, c], Self.Strides) + DataOffset);
    end
    else
    begin
      Index := OffsetToIndex(Self, Offset);
      Exit(IndexToStridedOffset(index, Self.Strides) {+ DataOffset})
    end;
  end;

  generic function TMultiArray.GetNDims: integer;
  begin
    Exit(Length(Shape));
  end;

  generic function TMultiArray.GetSize: longint;
  begin
    Exit(specialize Prod<longint>(Shape));
  end;

  function TMultiArray.GetVirtualData: TSingleVector;
  begin
    if Self.IsContiguous then Exit(Self.Data);
    Exit(multiarray.GetVirtualData(Self));
  end;

  function Im2Col(X: TMultiArray; FH, FW, Stride, Pad: longint): TMultiArray;
  var
    N, C, H, W: longint;
    NN, CC, HH, WW: longint;
    OutH, OutW: longint;
    col: TMultiArray;
    tthen: TDateTime;
  begin
    N := X.Shape[0];
    C := X.Shape[1];
    H := X.Shape[2];
    W := X.Shape[3];

    NN := X.Strides[0];
    CC := X.Strides[1];
    HH := X.Strides[2];
    WW := X.Strides[3];

    OutH := (H - FH) div Stride + 1;
    OutW := (W - FW) div Stride + 1;

    col := AsStrided(X, [N, OutH, OutW, C, FH, FW],
      [NN, Stride * HH, Stride * WW, CC, HH, WW]).Contiguous;
    Exit(col.Reshape([N * OutH * OutW, C * FH * FW]).T);
  end;

  function IndexOf(x: Single; A: TMultiArray): longint;
  var
    i: longint;
  begin
    Result := -1;
    for i := 0 to A.Size - 1 do
      if x = A.Get(i) then
        Exit(i);
  end;

  function IndexToStridedOffset(Index: array of longint; Strides: TLongVector): longint;
  var
    i: integer;
  begin
    Result := 0;
    for i := 0 to Length(Index) - 1 do
      Result := Result + (Strides[i] * Index[i]);
  end;

  function TMultiArray.Contiguous: TMultiArray;
  begin
    if IsContiguous then Exit(self);
    Exit(multiarray.Contiguous(Self));
  end;

  function TMultiArray.Copy(Deep: boolean = True): TMultiArray;
  var
    i, j: longint;
  begin
    Result.IsContiguous := self.IsContiguous;
    if deep then
    begin
      SetLength(Result.Data, length(Self.Data));
      for i := 0 to High(self.Data) do
        Result.Data[i] := self.Data[i];
    end
    else
      Result.Data := self.Data;
    Result.DataOffset := Self.DataOffset;
    Result.Shape := CopyVector(self.Shape);
    Result.Strides := CopyVector(self.Strides);

    for i := 0 to NDims - 1 do
    begin
      SetLength(Result.Indices, Length(Self.Indices));
      for j := 0 to Length(self.Indices[i]) - 1 do
      begin
        SetLength(Result.Indices[i], Length(self.Indices[i]));
        Result.Indices[i][j] := self.Indices[i][j];
      end;
    end;
  end;

  function TMultiArray.Get(i: longint): single; overload;
  begin
    if IsContiguous then Exit(Data[i]);
    Exit(Data[OffsetToStrided(i)]);
  end;

  function TMultiArray.Item: Single;
  begin
    if Size > 1 then raise Exception.Create('Can only convert 1-length array.');
    Exit(Data[0]);
  end;

  function TMultiArray.Matmul(Other: TMultiArray): TMultiArray;
  begin
    Exit(multiarray.Matmul(Self, Other));
  end;

  function TMultiArray.Reshape(NewShape: TLongVector): TMultiArray;
  var
    i: integer;
  begin
    if (specialize Prod<longint>(NewShape)) <> (specialize Prod<longint>(Shape)) then
      raise Exception.Create('Impossible reshape.');
    Result.Data := self.Data;
    Result.DataOffset := self.DataOffset;
    SetLength(Result.Shape, Length(NewShape));
    for i := 0 to Length(NewShape) - 1 do
      Result.Shape[i] := NewShape[i];
    Result.Strides := ShapeToStrides(Result.Shape);
    Result.ResetIndices;
  end;

  function TMultiArray.Slice(idx: array of TLongVector): TMultiArray;
  var
    i: integer;
  begin
    if Length(idx) > Self.NDims then
      raise Exception.Create('Invalid slicing: Len(idx) > Self.NDims');
    Result := Self.Copy(False);
    Result.IsContiguous := False;
    for i := 0 to High(idx) do
    begin
      if Length(idx[i]) > 0 then
      begin
        Result.Indices[i] := idx[i];
        Result.Shape[i] := Length(idx[i]);
      end
    end;
    Result := Result.Contiguous;
  end;

  function TMultiArray.SliceBool(BoolIdx: array of TLongVector): TMultiArray;
  var
    i, j, Cnt: longint;
    NewIdx: array of TLongVector;
  begin
    SetLength(NewIdx, Length(BoolIdx));
    for i := 0 to High(BoolIdx) do
    begin
      if Length(BoolIdx[i]) <> Shape[i] then
        raise Exception.Create('Length of BoolIdx[' + IntToStr(i) +
                               '] <> Array''s Shape[' + IntToStr(i) + ']');

      Cnt := 0;
      SetLength(NewIdx[i], Length(BoolIdx[i]));
      for j := 0 to High(BoolIdx[i]) do
      begin
        if BoolIdx[i][j] > 0 then
        begin
          NewIdx[i][Cnt] := j;
          Inc(Cnt);
        end;
      end;
      SetLength(NewIdx[i], Cnt);
    end;
    Exit(self[NewIdx].Contiguous);
  end;

  function TMultiArray.Squeeze: TMultiArray;
  begin
    Result := Self.Copy();
    SqueezeMultiArray(Result);
  end;

  function TMultiArray.T: TMultiArray;
  begin
    Exit(Transpose(Self));
  end;

  procedure TMultiArray.Put(i: array of integer; x: single);
  begin
    Data[IndexToStridedOffset(i, Self.Strides)] := x;
  end;

  procedure TMultiArray.ResetIndices;
  var
    i, j: integer;
  begin
    SetLength(Indices, NDims);
    for i := 0 to NDims - 1 do
    begin
      SetLength(Indices[i], Shape[i]);
      for j := 0 to Shape[i] - 1 do
        Indices[i][j] := j;
    end;
  end;

  { TNDIter }

  function TNDIter.Next(X: TMultiArray): single;
  begin
    Advance(X);
    Exit(X.Data[CurrentOffset]);
  end;

  procedure TNDIter.Advance(X: TMultiArray);
  var
    i, j, sum: longint;
  begin
    sum := 0;
    for i := 0 to NDims - 1 do
    begin
      Inc(sum, Strides[i] * X.Indices[i][CurrentVirt[i]]);
    end;
    CurrentOffset := sum;

    Inc(CurrentVirt[NDims - 1]);
    for i := NDims - 1 downto 1 do
    begin
      if CurrentVirt[i] >= Shape[i] then
      begin
        CurrentVirt[i] := 0;
        Inc(CurrentVirt[i - 1]);
      end;
    end;
  end;

  procedure TNDIter.Reset(X: TMultiArray);
  var
    i: integer;
  begin
    CurrentOffset := 0;
    NDims := X.NDims;
    Strides := X.Strides;
    Shape := X.Shape;
    SetLength(CurrentVirt, X.NDims);
    for i := 0 to High(CurrentVirt) do
      CurrentVirt[i] := 0;
  end;

  function _Add(a, b: single): single; inline;
  begin
    Exit(a + b);
  end;

  function _Divide(a, b: single): single; inline;
  begin
    Exit(a / b);
  end;

  function _EqualsTo(a, b: single): single; inline;
  begin
    Exit(Single(Integer(a = b)));
  end;

  function _GreaterThan(a, b: single): single; inline;
  begin
    Exit(Single(Integer(a > b)));
  end;

  function _GreaterEqualThan(a, b: single): single; inline;
  begin
    Exit(Single(Integer(a >= b)));
  end;

  function _LessThan(a, b: single): single; inline;
  begin
    Exit(Single(Integer(a < b)));
  end;

  function _LessEqualThan(a, b: single): single; inline;
  begin
    Exit(Single(Integer(a <= b)));
  end;

  function _Max(a, b: single): single; inline;
  begin
    Exit(math.max(a, b))
  end;

  function _Min(a, b: single): single; inline;
  begin
    Exit(math.min(a, b))
  end;

  function _Multiply(a, b: single): single; inline;
  begin
    Exit(a * b);
  end;

  function _Negate(a: single; params: array of single): single;
  begin
    Exit(-a);
  end;

  function _Power(base, exponent: single): single; inline;
  begin
    Exit(Power(base, exponent));
  end;

  function _Subtract(a, b: single): single; inline;
  begin
    Exit(a - b);
  end;

  function Add(A, B: TMultiArray): TMultiArray;
  begin
    Exit(ApplyBFunc(A, B, @_Add, GLOBAL_FUNC_DEBUG, 'ADD'));
  end;

  function Matmul(A, B: TMultiArray): TMultiArray;
  begin
    if (A.NDims <> 2) or (B.NDims <> 2) then
      raise Exception.Create('Only arrays with NDims=2 are supporter for Matmul');
    if (A.Shape[1] <> B.Shape[0]) then
      raise Exception.Create('A.Shape[1] must be equal to B.Shape[0]');
    Exit(MatMul_BACKEND(A, B));
  end;

  function Maximum(A, B: TMultiArray): TMultiArray;
  begin
    Exit(ApplyBFunc(A, B, @_Max, GLOBAL_FUNC_DEBUG, 'MAXIMUM'));
  end;

  function Minimum(A, B: TMultiArray): TMultiArray;
  begin
    Exit(ApplyBFunc(A, B, @_Min, GLOBAL_FUNC_DEBUG, 'MINIMUM'));
  end;

  function Multiply(A, B: TMultiArray): TMultiArray;
  begin
    Exit(ApplyBFunc(A, B, @_Multiply, GLOBAL_FUNC_DEBUG, 'MULTIPLY'));
  end;

  function Power(A, B: TMultiArray): TMultiArray;
  begin
    Exit(ApplyBFunc(A, B, @_Power));
  end;

  operator in (A: single; B: TSingleVector) C: boolean;
  var
    i: integer;
  begin
    C := False;
    for i := 0 to High(B) do
      if A = B[i] then
      begin
        C := True;
        Exit;
      end;
  end;

  operator +(A, B: TMultiArray) C: TMultiArray;
  begin
    C := Add(A, B);
  end;

  operator - (A: TMultiArray) B: TMultiArray;
  begin
    B := ApplyUFunc(A, @_Negate, []);
  end;

  operator - (A, B: TMultiArray) C: TMultiArray;
  begin
    C := ApplyBFunc(A, B, @_Subtract);
  end;

  operator *(A, B: TMultiArray) C: TMultiArray;
  begin
    C := Multiply(A, B);
  end;

  operator / (A, B: TMultiArray) C: TMultiArray;
  begin
    C := ApplyBFunc(A, B, @_Divide, GLOBAL_FUNC_DEBUG, 'DIVIDE');
  end;

  operator ** (A, B: TMultiArray) C: TMultiArray;
  begin
    C := Power(A, B);
  end;

  operator = (A, B: TMultiArray) C: TMultiArray;
  begin
    C := ApplyBFunc(A, B, @_EqualsTo, GLOBAL_FUNC_DEBUG, 'EQUALS_TO');
  end;

  operator > (A, B: TMultiArray) C: TMultiArray;
  begin
    C := ApplyBFunc(A, B, @_GreaterThan, GLOBAL_FUNC_DEBUG, 'GREATER_THAN');
  end;

  operator >= (A, B: TMultiArray) C: TMultiArray;
  begin
    C := ApplyBFunc(A, B, @_GreaterEqualThan, GLOBAL_FUNC_DEBUG, 'GREATER_EQUAL_THAN');
  end;

  operator<(A, B: TMultiArray)C: TMultiArray;
  begin
    C := ApplyBFunc(A, B, @_LessThan, GLOBAL_FUNC_DEBUG, 'LESS_THAN');
  end;

  operator<=(A, B: TMultiArray)C: TMultiArray;
  begin
    C := ApplyBFunc(A, B, @_LessEqualThan, GLOBAL_FUNC_DEBUG, 'LESS_EQUAL_THAN');
  end;

  operator in (A: TMultiArray; B: array of TMultiArray) C: boolean;
  var
    X: TMultiArray;
  begin
    C := False;
    for X in B do
      if ArrayEqual(A, X) then
      begin
        C := True;
        Exit;
      end;
  end;

  operator :=(A: single) B: TMultiArray;
  begin
    B := CreateMultiArray(A);
    SqueezeMultiArray(B);
  end;

  operator := (A: array of single) B: TMultiArray;
  begin
    B := CreateMultiArray(A);
  end;

  operator explicit(A: single) B: TMultiArray;
  begin
    B := A;
  end;

  operator :=(A: array of longint) B: TLongVector;
  begin
    B := DynArrayToVector(A);
  end;

  operator explicit(A: array of longint) B: TLongVector;
  begin
    B := A;
  end;

  operator :=(A: TLongVector) B: TSingleVector;
  var
    i: longint;
  begin
    SetLength(B, Length(A));
    for i := 0 to High(A) do B[i] := A[i];
  end;

  operator explicit(A: TLongVector) B: TSingleVector;
  begin
    B := A;
  end;

  operator :=(A: TSingleVector) B: TLongVector;
  var
    i: longint;
  begin
    SetLength(B, Length(A));
    for i := 0 to High(A) do B[i] := Round(A[i]);
  end;

  operator explicit(A: TSingleVector) B: TLongVector;
  begin
    B := A;
  end;

  operator :=(A: longint) B: TLongVector;
  begin
    SetLength(B, 1);
    B[0] := A;
  end;

  operator explicit(A: longint) B: TLongVector;
  begin
    B := A;
  end;

  operator :=(A: TMultiArray) B: TSingleVector;
  begin
    if A.Squeeze.NDims > 1 then
      raise Exception.Create('Can only convert MultiArray with NDims=1');
    B := A.GetVirtualData;
  end;

  operator explicit(A: TMultiArray) B: TSingleVector;
  begin
    B := A;
  end;

  operator :=(A: TMultiArray) B: TLongVector;
  begin
    if A.Squeeze.NDims > 1 then
      raise Exception.Create('Can only convert MultiArray with NDims=1');
    B := A.GetVirtualData;
  end;

  operator explicit(A: TMultiArray) B: TLongVector;
  begin
    B := A;
  end;

initialization
  GLOBAL_FUNC_DEBUG := False;

end.

