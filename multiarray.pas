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
  TMultiArray = record
  private
    FIsContiguous: boolean;
    function GetNDims: integer;
    function GetSize: longint;
    function IndexToStridedOffset(Index: array of longint): longint;
    function OffsetToStrided(Offset: longint): longint; inline;
  public
    FDataOffset: longint;
    Data:  array of single;
    Indices: TLongVectorArr;
    Shape: TLongVector;
    Strides: TLongVector;
    function Contiguous: TMultiArray;
    function Copy(Deep: boolean = True): TMultiArray;
    function Get(i: longint): single; overload;
    function Get(idx: array of integer): TMultiArray; overload;
    function Reshape(NewShape: array of longint): TMultiArray;
    { For now Slice() will return contiguous }
    function Slice(idx: array of TLongVector): TMultiArray;
    function T: TMultiArray;
    procedure Put(i: array of integer; x: single);
    procedure ResetIndices;
    property IsContiguous: boolean read FIsContiguous write FIsContiguous;
    property NDims: integer read GetNDims;
    property Size: longint read GetSize;
  end;

  TBroadcastResult = record
    A, B: TMultiArray;
  end;

  function All: TLongVector;
  function AllocateMultiArray(Size: longint): TMultiArray;
  function ApplyBFunc(A, B: TMultiArray; BFunc: TBFunc; PrintDebug: Boolean = False;
    FuncName: string = ''): TMultiArray;
  function ApplyUFunc(A: TMultiArray; UFunc: TUFunc; Params: array of single): TMultiArray;
  function AsStrided(A: TMultiArray; Shape, Strides: array of longint): TMultiArray;
  function BroadcastArrays(A, B: TMultiArray): TBroadcastResult;
  function CreateEmptyFTensor(Contiguous: boolean = True): TMultiArray;
  function CreateMultiArray(AData: array of single): TMultiArray;
  function CreateMultiArray(AData: single): TMultiArray;
  function DynArrayToVector(A: array of longint): TLongVector;
  function GenerateMultiArray(Shape: array of longint; GenFunc: TGenFunc;
    Params: array of single): TMultiArray;
  function GetVirtualData(A: TMultiArray): TSingleVector;
  function Range(Start, Stop, step: longint): TLongVector; overload;
  function Range(Start, Stop: longint): TLongVector; overload;
  function ShapeToStrides(AShape: array of longint): TLongVector;

  procedure DebugMultiArray(A: TMultiArray);
  procedure PrintMultiArray(A: TMultiArray);
  procedure SqueezeMultiArray(var A: TMultiArray);

  { Vector tools }
  generic function CopyVector<T>(v: T): T;
  generic function SliceVector<T>(v: T; start, stop: longint): T;
  generic function Prod<T>(Data: array of T): T;
  generic function Reverse<T>(Data: T): T;
  generic function VectorsEqual<T>(A, B: T): boolean;
  generic procedure DeleteFromVector<T>(var Data: T; At, Count: longint);
  generic procedure PrintVector<T>(Data: T);

  { Basic arithmetic function wrappers }
  function _Add(a, b: single): single;
  function _Divide(a, b: single): single;
  function _Multiply(a, b: single): single;
  function _Negate(a: single; params: array of single): single;
  function _Power(base, exponent: single): single;
  function _Subtract(a, b: single): single;

  function Power(A, B: TMultiArray): TMultiArray; overload;
  operator + (A, B: TMultiArray) C: TMultiArray;
  operator - (A: TMultiArray) B: TMultiArray;
  operator - (A, B: TMultiArray) C: TMultiArray;
  operator * (A, B: TMultiArray) C: TMultiArray;
  operator / (A, B: TMultiArray) C: TMultiArray;
  operator ** (A, B: TMultiArray) C: TMultiArray;
  operator := (A: single) B: TMultiArray;
  operator := (A: array of single) B: TMultiArray;
  operator explicit(A: single) B: TMultiArray;

  operator :=(A: array of longint) B: TLongVector;
  operator explicit(A: array of longint) B: TLongVector;
  operator :=(A: TLongVector) B: TSingleVector;
  operator explicit(A: TLongVector) B: TSingleVector;

var
  GLOBAL_FUNC_DEBUG: boolean;

implementation

  function All: TLongVector;
  begin
    Exit([]);
  end;

  function AllocateMultiArray(Size: longint): TMultiArray;
  begin
    SetLength(Result.Data, Size);
    SetLength(Result.Shape, 1);
    Result.IsContiguous := True;
    Result.Shape[0] := Size;
    Result.Strides := ShapeToStrides(Result.Shape);
    Result.ResetIndices;
    Result.FDataOffset := 0;
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

  function CreateEmptyFTensor(Contiguous: boolean = True):TMultiArray;
  begin
    Result.IsContiguous := Contiguous;
    Result.FDataOffset := 0;
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
  end;

  function AsStrided(A: TMultiArray; Shape, Strides: array of longint): TMultiArray;
  begin
    Result := CreateEmptyFTensor(false);
    Result.Data := A.Data;
    Result.Shape := DynArrayToVector(Shape);
    Result.Strides := DynArrayToVector(Strides);
    Result.Indices := A.Indices;
  end;

  function ApplyBFunc(A, B: TMultiArray; BFunc: TBFunc; PrintDebug: Boolean = False;
    FuncName: string = ''): TMultiArray;
  var
    i: longint;
    BcastResult: TBroadcastResult;
    TimeThen: TDateTime;
  begin
    if (PrintDebug or GLOBAL_FUNC_DEBUG) then TimeThen := Now;

    if not(specialize VectorsEqual<TLongVector>(A.Shape, B.Shape)) then
    begin
      BcastResult := BroadcastArrays(A, B);
      Result := ApplyBFunc(BcastResult.A, BcastResult.B, BFunc, PrintDebug, FuncName);
    end else
    begin
      Result := AllocateMultiArray(A.Size);
      Result := Result.Reshape(A.Shape); // should be reset strides
      Result.ResetIndices;
      for i := 0 to A.Size - 1 do
        Result.Data[i] := BFunc(A.Get(i), B.Get(i));

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
    for i := 0 to High(A.Data) do
      Result.Data[i] := UFunc(Result.Data[i], Params);
    Result.ResetIndices;
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
    writeln('Data offset    :', A.FDataOffset);
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

  procedure PrintMatrix(A: TMultiArray);
  var
    r, c: longint;
  begin
    if A.NDims > 2 then
      raise Exception.Create('Cannot print array with NDims > 2');

    for r := 0 to A.Shape[0] - 1 do
    begin
      for c := 0 to A.Shape[1] - 1 do
      begin
        Write(A.Get(r * A.Shape[1] + c) : 5 : 2);
        if c < A.Shape[1] - 1 then Write(' ');
      end;

      if r < (A.Shape[0] - 1) then
        WriteLn;
    end;
  end;

  procedure PrintMultiArray(A: TMultiArray);
    procedure Pr(A: TMultiArray; Shape: TLongVector);
    var
      i: longint;
    begin
      if Length(Shape) < 2 then
      begin
        for i := 0 to A.Size - 1 do
          Write(A.Get(i) : 5 : 2);
      end
      else
      if Length(Shape) = 2 then
      begin
        PrintMatrix(A);
      end
      else
      begin
        for i := 0 to Shape[0] - 1 do
        begin
          Pr(A.Get([i]), specialize SliceVector<TLongVector>(Shape, 1, Length(Shape)));
        end;
      end;
      WriteLn(DupeString(sLineBreak, Length(Shape) - 1));
    end;
  begin
    Pr(A, A.Shape);
  end;

  generic function CopyVector<T>(v: T): T;
  var
    i: longint;
  begin
    SetLength(Result, Length(v));
    for i := 0 to High(v) do
      Result[i] := v[i];
  end;

  generic function SliceVector<T>(v: T; start, stop: longint): T;
  var
    i: longint;
  begin
    SetLength(Result, stop - start);
    for i := start to stop - 1 do
      Result[i - start] := v[i];
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

  generic function VectorsEqual<T>(A, B: T): boolean;
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
      Write(Data[i]);
      if i < Length(Data) - 1 then
        Write(', ');
    end;
    WriteLn;
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
    SetLength(Result, A.Size);
    for i := 0 to A.Size - 1 do
      Result[i] := A.Get(i);
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

  function ShapeToStrides(AShape: array of longint): TLongVector;
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

  procedure SqueezeMultiArray(var A: TMultiArray);
  var
    len, i: integer;
  begin
    len := A.NDims;
    for i := len - 1 downto 0 do
      if A.Shape[i] = 1 then
      begin
        specialize DeleteFromVector<TLongVector>(A.Shape, i, 1);
        specialize DeleteFromVector<TLongVector>(A.Strides, i, 1);
        specialize DeleteFromVector<TLongVectorArr>(A.Indices, i, 1);
      end;
  end;

  generic function TMultiArray.OffsetToStrided(Offset: longint): longint;
  var
    r, c, i, cnt: longint;
    index: TLongVector;
  begin
    if IsContiguous then Exit(Offset);

    { A scalar }
    if Length(Data) = 1 then Exit(0);

    { A vector }
    if NDims = 1 then Exit(Indices[0][Offset] + FDataOffset);

    { Higher rank }
    if NDims = 2 then
    begin
      r := Indices[0][Offset div Shape[1]];
      c := Indices[1][Offset mod Shape[1]];
      Exit(IndexToStridedOffset([r, c]) + FDataOffset);
    end
    else
    begin
      SetLength(index, Self.NDims);
      cnt := Self.NDims - 1;
      for i := High(Self.Shape) downto 0 do
      begin
        index[cnt] := Offset mod Self.Shape[i];
        Offset := Offset div Self.Shape[i];
        Dec(cnt);
      end;
      Exit(IndexToStridedOffset(index) + FDataOffset)
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

  function TMultiArray.IndexToStridedOffset(Index: array of longint): longint;
  var
    i: integer;
  begin
    Result := 0;
    for i := 0 to Length(Index) - 1 do
      Result := Result + (Strides[i] * Index[i]);
  end;

  function TMultiArray.Contiguous: TMultiArray;
  var
    i: integer;
  begin
    if IsContiguous then Exit(self);
    Result := AllocateMultiArray(Self.Size).Reshape(Self.Shape);
    Result.ResetIndices;
    Result.IsContiguous := True;
    SetLength(Result.Data, Self.Size);
    for i := 0 to Self.Size - 1 do
      Result.Data[i] := Self.Get(i);
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
    Result.FDataOffset := Self.FDataOffset;
    Result.Shape := specialize CopyVector<TLongVector>(self.Shape);
    Result.Strides := specialize CopyVector<TLongVector>(self.Strides);

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

  function TMultiArray.Get(idx: array of integer): TMultiArray;
  var
    i, Offset: longint;
    NewIdx, NewShape, NewStrides: TLongVector;
  begin
    if Length(idx) > Self.NDims then
      raise Exception.Create('Index out of bounds');

    if Self.NDims = 1 then
      Exit(Self.Get(idx[0]));

    if Length(idx) < Self.NDims then
    begin
      SetLength(NewIdx, Self.NDims);
      for i := 0 to High(idx) do
        NewIdx[i] := idx[i];

      SetLength(NewShape, Self.NDims - Length(idx));
      for i := Length(idx) to High(Self.Shape) do
        NewShape[i - Length(idx)] := Self.Shape[i];

      SetLength(NewStrides, Self.NDims - Length(idx));
      for i := Length(idx) to High(Self.Strides) do
        NewStrides[i - Length(idx)] := Self.Strides[i];
    end
    else
    begin
      NewIdx := Idx;
      NewShape := [1];
      NewStrides := [1];
    end;

    Offset := IndexToStridedOffset(NewIdx);

    Result := CreateEmptyFTensor(Self.IsContiguous);
    Result.Data := Self.Data;
    Result.Shape := NewShape;
    Result.Strides := NewStrides;
    Result.FDataOffset := Self.FDataOffset + Offset;
    Result.ResetIndices;
  end;

  function TMultiArray.Reshape(NewShape: array of longint): TMultiArray;
  var
    i: integer;
  begin
    Assert((specialize Prod<longint>(NewShape)) = (specialize Prod<longint>(Shape)), 'Impossible reshape.');
    Result.Data := self.Data;
    Result.FDataOffset := self.FDataOffset;
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
    Result.Strides := ShapeToStrides(Result.Shape);
    Result := Result.Contiguous;
  end;

  function TMultiArray.T: TMultiArray;
  var
    i: integer;
    NewIndices: array of array of longint;
  begin
    Result := Self.Copy();
    Result.FIsContiguous := False;
    Result.Shape := specialize Reverse<TLongVector>(Self.Shape);
    Result.Strides := specialize Reverse<TLongVector>(Self.Strides);
    SetLength(NewIndices, Length(Result.Indices));
    for i := 0 to High(Result.Indices) do
      NewIndices[i] := Result.Indices[High(Result.Indices) - i];
    Result.Indices := NewIndices;
  end;

  procedure TMultiArray.Put(i: array of integer; x: single);
  begin
    Data[Self.IndexToStridedOffset(i)] := x;
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

  function _Add(a, b: single): single;
  begin
    Exit(a + b);
  end;

  function _Divide(a, b: single): single;
  begin
    Exit(a / b);
  end;

  function _Multiply(a, b: single): single;
  begin
    Exit(a * b);
  end;

  function _Negate(a: single; params: array of single): single;
  begin
    Exit(-a);
  end;

  function _Power(base, exponent: single): single;
  begin
    Exit(Power(base, exponent));
  end;

  function _Subtract(a, b: single): single;
  begin
    Exit(a - b);
  end;

  function Power(A, B: TMultiArray): TMultiArray;
  begin
    Exit(ApplyBFunc(A, B, @_Power));
  end;

  operator +(A, B: TMultiArray) C: TMultiArray;
  begin
    C := ApplyBFunc(A, B, @_Add, GLOBAL_FUNC_DEBUG, 'ADD');
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
    C := ApplyBFunc(A, B, @_Multiply);
  end;

  operator / (A, B: TMultiArray) C: TMultiArray;
  begin
    C := ApplyBFunc(A, B, @_Divide, GLOBAL_FUNC_DEBUG, 'DIVIDE');
  end;

  operator ** (A, B: TMultiArray) C: TMultiArray;
  begin
    C := Power(A, B);
  end;

  operator :=(A: single) B: TMultiArray;
  begin
    B := CreateMultiArray(A);
  end;

  operator := (A: array of single) B: TMultiArray;
  begin
    B := CreateMultiArray(A);
  end;

  operator explicit(A: single) B: TMultiArray;
  begin
    B := CreateMultiArray(A);
  end;

  operator :=(A: array of longint) B: TLongVector;
  begin
    B := DynArrayToVector(A);
  end;

  operator explicit(A: array of longint) B: TLongVector;
  begin
    B := DynArrayToVector(A);
  end;

  operator :=(A: TLongVector) B: TSingleVector;
  var
    i: longint;
  begin
    SetLength(B, Length(A));
    for i := 0 to High(A) do B[i] := A[i];
  end;

  operator explicit(A: TLongVector) B: TSingleVector;
  var
    i: longint;
  begin
    SetLength(B, Length(A));
    for i := 0 to High(A) do B[i] := A[i];
  end;

initialization
  GLOBAL_FUNC_DEBUG := False;

end.

